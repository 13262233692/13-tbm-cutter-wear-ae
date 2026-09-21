"""AE 信号采集模块。

模拟刀盘上 8 个声发射传感器的高频波形采集（采样率 1 MHz），
并将原始波形连同元数据写入 HDF5 文件。

实际部署时可将 `simulate_burst_signal` 替换为真实 DAQ 采集卡的读取逻辑，
HDF5 数据模型保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

SAMPLING_RATE = 1_000_000  # 1 MHz
NUM_SENSORS = 8

# 8 个传感器在刀盘上的 (半径 m, 角度 deg) 布置
SENSOR_LAYOUT = [
    (0.5, 0), (0.5, 90), (0.5, 180), (0.5, 270),
    (1.2, 45), (1.2, 135), (1.2, 225), (1.2, 315),
]


@dataclass
class AcquisitionConfig:
    sampling_rate: int = SAMPLING_RATE
    num_sensors: int = NUM_SENSORS
    window_seconds: float = 0.02  # 每个采集窗口 20 ms
    wear_stage: float = 0.0       # 0.0 新刀 -> 1.0 完全磨损
    seed: int | None = None
    sensor_layout: list = field(default_factory=lambda: list(SENSOR_LAYOUT))


def simulate_burst_signal(config: AcquisitionConfig, sensor_idx: int,
                          rng: np.random.Generator) -> np.ndarray:
    """模拟单个传感器一个窗口的 AE 波形。

    磨损加剧时：突发事件更频繁、幅值更大、主频向低频偏移、
    背景噪声增强——用于驱动特征趋势。
    """
    n = int(config.sampling_rate * config.window_seconds)
    t = np.arange(n) / config.sampling_rate
    stage = float(np.clip(config.wear_stage, 0.0, 1.0))

    # 背景噪声随磨损增强
    noise_std = 0.05 + 0.15 * stage
    signal = rng.normal(0.0, noise_std, n)

    # AE 突发：衰减正弦波包络
    n_bursts = rng.integers(2, 4 + int(6 * stage))
    for _ in range(n_bursts):
        center = rng.uniform(0.001, config.window_seconds - 0.002)
        freq = rng.uniform(120e3, 300e3) * (1.0 - 0.45 * stage)  # 主频随磨损下移
        decay = rng.uniform(20e3, 80e3)
        amp = rng.uniform(0.3, 1.0) * (0.5 + 1.8 * stage)
        env = np.exp(-decay * np.maximum(t - center, 0.0)) * (t >= center)
        signal += amp * env * np.sin(2 * np.pi * freq * (t - center))

    # 传感器位置引入轻微通道间差异
    signal *= 1.0 + 0.05 * np.sin(sensor_idx)
    return signal.astype(np.float32)


def acquire_window(config: AcquisitionConfig) -> np.ndarray:
    """采集一个窗口内全部传感器的波形，返回 (num_sensors, n_samples)。"""
    rng = np.random.default_rng(config.seed)
    return np.stack([
        simulate_burst_signal(config, i, rng) for i in range(config.num_sensors)
    ])


def save_hdf5(path: str, waveforms: np.ndarray, config: AcquisitionConfig,
              timestamp: str) -> None:
    """将波形数据与元数据写入 HDF5。"""
    with h5py.File(path, "w") as h5:
        h5.attrs["sampling_rate"] = config.sampling_rate
        h5.attrs["num_sensors"] = config.num_sensors
        h5.attrs["window_seconds"] = config.window_seconds
        h5.attrs["wear_stage"] = config.wear_stage
        h5.attrs["timestamp"] = timestamp
        layout = np.asarray(config.sensor_layout, dtype=np.float64)
        h5.create_dataset("sensor_layout", data=layout)  # (n, 2): radius, angle_deg
        h5.create_dataset(
            "waveforms", data=waveforms,
            compression="gzip", compression_opts=4,
        )


def load_hdf5(path: str) -> tuple[np.ndarray, dict, np.ndarray]:
    """读取 HDF5，返回 (waveforms, attrs, sensor_layout)。"""
    with h5py.File(path, "r") as h5:
        waveforms = h5["waveforms"][:]
        layout = h5["sensor_layout"][:]
        attrs = dict(h5.attrs.items())
    return waveforms, attrs, layout
