"""特征提取模块.

对声发射波形进行小波阈值降噪, 并提取磨损敏感特征:
  - RMS (均方根): 反映信号能量, 磨损加剧时上升
  - 峰度 (Kurtosis): 反映冲击成分, 对突发型磨损敏感
  - 频谱质心 (Spectral Centroid): 反映频谱重心迁移
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy import stats


def wavelet_denoise(signal: np.ndarray, wavelet: str = "db6",
                    level: int | None = None) -> np.ndarray:
    """小波软阈值降噪 (VisuShrink 通用阈值 + 分层噪声估计).

    对每层细节系数使用 median(|cD|)/0.6745 估计噪声标准差,
    采用 sigma*sqrt(2*ln(N)) 阈值进行软阈值收缩后重构.
    """
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        level = min(level, 6)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    n = len(signal)
    denoised = [coeffs[0]]
    for detail in coeffs[1:]:
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * np.sqrt(2.0 * np.log(n))
        denoised.append(pywt.threshold(detail, threshold, mode="soft"))
    reconstructed = pywt.waverec(denoised, wavelet)
    return reconstructed[:n]


def compute_rms(signal: np.ndarray) -> float:
    """均方根值."""
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_kurtosis(signal: np.ndarray) -> float:
    """峰度 (Fisher 定义, 正态分布为 0)."""
    return float(stats.kurtosis(signal, fisher=True))


def compute_spectral_centroid(signal: np.ndarray,
                              sampling_rate: int) -> float:
    """频谱质心 (Hz)."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


@dataclass
class ChannelFeatures:
    """单通道特征."""

    channel: int
    rms: float
    kurtosis: float
    spectral_centroid: float


@dataclass
class WindowFeatures:
    """一个采集窗口全部通道的特征汇总."""

    window_index: int
    timestamp: float
    channels: list

    def mean(self, name: str) -> float:
        return float(np.mean([getattr(c, name) for c in self.channels]))


class FeatureExtractor:
    """特征提取器: 小波降噪 + 三特征提取."""

    def __init__(self, sampling_rate: int, wavelet: str = "db6",
                 denoise: bool = True):
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.denoise = denoise

    def process_window(self, waveforms: np.ndarray, window_index: int,
                       timestamp: float) -> WindowFeatures:
        """处理一个采集窗口 (num_sensors, n_samples)."""
        channels = []
        for ch in range(waveforms.shape[0]):
            signal = waveforms[ch]
            if self.denoise:
                signal = wavelet_denoise(signal, wavelet=self.wavelet)
            channels.append(ChannelFeatures(
                channel=ch,
                rms=compute_rms(signal),
                kurtosis=compute_kurtosis(signal),
                spectral_centroid=compute_spectral_centroid(
                    signal, self.sampling_rate),
            ))
        return WindowFeatures(
            window_index=window_index,
            timestamp=timestamp,
            channels=channels,
        )
