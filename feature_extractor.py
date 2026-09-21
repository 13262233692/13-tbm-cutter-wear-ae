"""特征提取模块：小波降噪 + RMS / 峰度 / 频谱质心。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy import stats


@dataclass
class DenoiseConfig:
    wavelet: str = "db6"
    level: int = 5
    threshold_mode: str = "soft"


def wavelet_denoise(signal: np.ndarray, config: DenoiseConfig | None = None) -> np.ndarray:
    """小波阈值降噪（VisuShrink 通用阈值 + 软阈值）。"""
    config = config or DenoiseConfig()
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(config.wavelet).dec_len)
    level = min(config.level, max_level)
    coeffs = pywt.wavedec(signal, config.wavelet, level=level)

    # 由最细尺度细节系数估计噪声标准差（中位数绝对偏差）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(len(signal)))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=config.threshold_mode)
        for c in coeffs[1:]
    ]
    out = pywt.waverec(denoised_coeffs, config.wavelet)
    return out[: len(signal)]


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def compute_kurtosis(signal: np.ndarray) -> float:
    return float(stats.kurtosis(signal, fisher=False))


def compute_spectral_centroid(signal: np.ndarray, sampling_rate: float) -> float:
    """频谱质心 (Hz)：幅度谱的加权平均频率。"""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def extract_features(signal: np.ndarray, sampling_rate: float,
                     denoise: bool = True,
                     denoise_config: DenoiseConfig | None = None) -> dict:
    """对单通道信号提取特征，返回特征字典。"""
    clean = wavelet_denoise(signal, denoise_config) if denoise else np.asarray(signal)
    return {
        "rms": compute_rms(clean),
        "kurtosis": compute_kurtosis(clean),
        "spectral_centroid": compute_spectral_centroid(clean, sampling_rate),
    }


def extract_all_channels(waveforms: np.ndarray, sampling_rate: float,
                         denoise: bool = True) -> list[dict]:
    """对全部传感器通道提取特征。"""
    return [extract_features(ch, sampling_rate, denoise) for ch in waveforms]
