"""报告生成模块：JSON 报告 + 特征趋势可视化。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURE_KEYS = ["rms", "kurtosis", "spectral_centroid"]
FEATURE_LABELS = {
    "rms": "RMS",
    "kurtosis": "Kurtosis",
    "spectral_centroid": "Spectral Centroid (Hz)",
}


def plot_feature_trends(times: np.ndarray,
                        feature_history: list[list[dict]],
                        out_path: str) -> str:
    """绘制全部传感器的 RMS / 峰度 / 频谱质心随时间趋势。"""
    n_sensors = len(feature_history)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for key, ax in zip(FEATURE_KEYS, axes):
        for s in range(n_sensors):
            values = [f[key] for f in feature_history[s]]
            ax.plot(times, values, marker=".", ms=3, lw=1, label=f"S{s + 1}")
        ax.set_ylabel(FEATURE_LABELS[key])
        ax.grid(alpha=0.3)
    axes[0].set_title("AE Feature Trends (8 sensors)")
    axes[0].legend(ncol=4, fontsize=8, loc="best")
    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_di_and_weibull(times: np.ndarray, di_series: np.ndarray,
                        weibull: dict, failure_threshold: float,
                        out_path: str, sensor_name: str = "") -> str:
    """绘制退化指数趋势与威布尔拟合曲线、失效阈值线。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, di_series, "o-", ms=4, label="Degradation Index")

    beta, eta = weibull.get("beta"), weibull.get("eta")
    if beta and eta:
        t_max = max(times[-1], eta * (-np.log(1 - failure_threshold)) ** (1 / beta))
        t_fit = np.linspace(max(times[0], 1e-3), t_max * 1.05, 300)
        cdf = 1.0 - np.exp(-((t_fit / eta) ** beta))
        ax.plot(t_fit, cdf, "--", label=f"Weibull fit (β={beta:.2f}, η={eta:.1f})")

    ax.axhline(failure_threshold, color="r", ls=":", label=f"Failure threshold {failure_threshold}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Degradation Index")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Wear Degradation & Weibull RUL {sensor_name}".strip())
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_denoise_comparison(raw: np.ndarray, denoised: np.ndarray,
                            sampling_rate: float, out_path: str,
                            sensor_name: str = "") -> str:
    """小波降噪前后波形与频谱对比。"""
    t = np.arange(len(raw)) / sampling_rate * 1e3  # ms
    freqs = np.fft.rfftfreq(len(raw), 1.0 / sampling_rate) / 1e3  # kHz

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (sig, name) in enumerate([(raw, "Raw"), (denoised, "Denoised")]):
        axes[0, col].plot(t, sig, lw=0.4)
        axes[0, col].set_title(f"{name} waveform {sensor_name}".strip())
        axes[0, col].set_xlabel("Time (ms)")
        spec = np.abs(np.fft.rfft(sig))
        axes[1, col].plot(freqs, spec, lw=0.6)
        axes[1, col].set_title(f"{name} spectrum")
        axes[1, col].set_xlabel("Frequency (kHz)")
        axes[1, col].set_xlim(0, sampling_rate / 2 / 1e3)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_report(sensor_results: list[dict], sensor_layout: np.ndarray,
                 times: np.ndarray, figures: dict, meta: dict) -> dict:
    """汇总全部传感器评估结果，生成 JSON 兼容的报告字典。"""
    sensors = []
    for idx, res in enumerate(sensor_results):
        radius, angle = sensor_layout[idx]
        sensors.append({
            "sensor_id": f"S{idx + 1}",
            "position": {"radius_m": float(radius), "angle_deg": float(angle)},
            "current_di": res["current_di"],
            "wear_state": res["wear_state"],
            "wear_state_label": res["wear_state_label"],
            "weibull": res["weibull"],
            "rul_hours": res["rul"]["rul"],
            "predicted_failure_time_h": res["rul"]["failure_time"],
        })

    valid_ruls = [s["rul_hours"] for s in sensors if s["rul_hours"] is not None]
    worst = max(sensors, key=lambda s: s["current_di"])
    return {
        "meta": meta,
        "time_range_h": [float(times[0]), float(times[-1])],
        "overall": {
            "worst_sensor": worst["sensor_id"],
            "worst_di": worst["current_di"],
            "wear_state": worst["wear_state"],
            "wear_state_label": worst["wear_state_label"],
            "min_rul_hours": min(valid_ruls) if valid_ruls else None,
        },
        "sensors": sensors,
        "figures": figures,
    }


def write_json_report(report: dict, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    return out_path
