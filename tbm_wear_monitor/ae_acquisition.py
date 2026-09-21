"""声发射信号采集模块.

负责 8 通道声发射传感器的高频波形采集 (采样率 1 MHz),
并将原始波形与采集元数据写入 HDF5 文件.

在无真实采集硬件的环境下, 内置物理一致的仿真信号发生器:
随着刀具磨损加剧, 声发射事件的幅值、事件率与高频成分上升,
用于端到端验证监测与寿命预测流程.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import h5py
import numpy as np

SAMPLING_RATE = 1_000_000  # 1 MHz
NUM_SENSORS = 8


@dataclass
class AcquisitionConfig:
    """采集参数配置."""

    sampling_rate: int = SAMPLING_RATE
    num_sensors: int = NUM_SENSORS
    window_seconds: float = 0.05  # 每个采集窗口时长 (s)
    sensor_positions: list = field(
        default_factory=lambda: [
            f"cutter_ring_{i + 1}" for i in range(NUM_SENSORS)
        ]
    )

    @property
    def samples_per_window(self) -> int:
        return int(self.sampling_rate * self.window_seconds)


class AEWaveformSimulator:
    """声发射波形仿真器.

    信号模型: 背景高斯噪声 + 泊松到达的指数衰减振荡突发 (burst).
    磨损度 wear_ratio in [0, 1] 控制突发幅值、事件率与中心频率.
    """

    def __init__(self, config: AcquisitionConfig, seed: int | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def generate_window(self, wear_ratio: float) -> np.ndarray:
        """生成一个采集窗口的 8 通道波形, 返回 (num_sensors, n_samples)."""
        n = self.cfg.samples_per_window
        fs = self.cfg.sampling_rate
        t = np.arange(n) / fs
        waveforms = np.empty((self.cfg.num_sensors, n), dtype=np.float64)

        for ch in range(self.cfg.num_sensors):
            # 背景噪声水平随磨损缓慢上升
            noise_std = 0.05 * (1.0 + 0.8 * wear_ratio)
            signal = self.rng.normal(0.0, noise_std, n)

            # 突发事件率: 新刀约 150 次/s, 严重磨损可达 650 次/s
            event_rate = 150.0 + 500.0 * wear_ratio
            n_events = self.rng.poisson(event_rate * self.cfg.window_seconds)
            for _ in range(n_events):
                start = int(self.rng.integers(0, n))
                # 幅值随磨损显著增大; 高磨损时出现少量强冲击事件,
                # 使幅值分布呈重尾 (峰度上升)
                amplitude = self.rng.uniform(0.2, 1.0) * (0.5 + 2.5 * wear_ratio)
                if self.rng.random() < 0.05 + 0.30 * wear_ratio:
                    amplitude *= 1.0 + 6.0 * wear_ratio
                # 中心频率 80kHz ~ 320kHz, 磨损后高频成分增多
                freq = self.rng.uniform(80e3, 180e3 + 140e3 * wear_ratio)
                decay = self.rng.uniform(20e-6, 80e-6)
                length = min(n - start, int(decay * 8 * fs))
                if length <= 0:
                    continue
                tt = t[:length]
                burst = amplitude * np.exp(-tt / decay) * np.sin(
                    2 * np.pi * freq * tt
                )
                signal[start : start + length] += burst
            waveforms[ch] = signal
        return waveforms


class AEAcquisition:
    """声发射采集器: 采集波形并写入 HDF5."""

    def __init__(self, config: AcquisitionConfig | None = None,
                 simulator: AEWaveformSimulator | None = None):
        self.cfg = config or AcquisitionConfig()
        self.simulator = simulator or AEWaveformSimulator(self.cfg)

    def acquire(self, wear_ratio: float) -> np.ndarray:
        """采集一个窗口的波形数据."""
        return self.simulator.generate_window(wear_ratio)

    def save_window(self, h5file: h5py.File, window_index: int,
                    timestamp: float, wear_ratio: float,
                    waveforms: np.ndarray) -> None:
        """将一个采集窗口写入 HDF5 组 /windows/<index>."""
        grp = h5file.create_group(f"windows/{window_index:05d}")
        grp.create_dataset(
            "waveforms", data=waveforms.astype(np.float32),
            compression="gzip", compression_opts=4,
        )
        grp.attrs["timestamp"] = timestamp
        grp.attrs["wear_ratio"] = wear_ratio
        grp.attrs["sampling_rate"] = self.cfg.sampling_rate
        grp.attrs["num_sensors"] = self.cfg.num_sensors


def open_acquisition_file(path: str, config: AcquisitionConfig) -> h5py.File:
    """创建 HDF5 采集文件并写入全局元数据."""
    h5f = h5py.File(path, "w")
    h5f.attrs["created_at"] = _dt.datetime.now().isoformat()
    h5f.attrs["sampling_rate"] = config.sampling_rate
    h5f.attrs["num_sensors"] = config.num_sensors
    h5f.attrs["window_seconds"] = config.window_seconds
    h5f.attrs["sensor_positions"] = list(config.sensor_positions)
    h5f.require_group("windows")
    return h5f


def load_window(h5file: h5py.File, window_index: int) -> tuple[np.ndarray, dict]:
    """从 HDF5 读取一个窗口的波形与属性."""
    grp = h5file[f"windows/{window_index:05d}"]
    waveforms = grp["waveforms"][:].astype(np.float64)
    attrs = {k: grp.attrs[k] for k in grp.attrs}
    return waveforms, attrs
