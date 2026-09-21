"""报告生成模块.

输出:
  - JSON 监测报告 (磨损状态、特征趋势摘要、剩余寿命预测)
  - PNG 趋势图 (RMS / 峰度 / 频谱质心 / 退化指标随时间变化)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .wear_model import STAGE_THRESHOLDS, WeibullRUL

# 图内标注使用英文, 避免运行环境缺少中文字体
STAGE_LABELS_EN = ["Normal", "Early wear", "Moderate wear", "Severe wear"]


def generate_json_report(path: str, assessments: list,
                         rul: WeibullRUL,
                         feature_history: dict,
                         num_sensors: int) -> dict:
    """生成 JSON 监测报告并写入磁盘."""
    latest = assessments[-1]
    report = {
        "report_type": "tbm_cutter_wear_monitoring",
        "num_sensors": num_sensors,
        "num_windows": len(assessments),
        "current_status": {
            "timestamp": latest.timestamp,
            "degradation_index": round(latest.degradation_index, 4),
            "wear_stage": latest.stage,
            "wear_stage_index": latest.stage_index,
        },
        "feature_summary": {
            name: {
                "latest": round(float(values[-1]), 4),
                "mean": round(float(np.mean(values)), 4),
                "max": round(float(np.max(values)), 4),
            }
            for name, values in feature_history.items()
        },
        "rul_prediction": {
            "model": "two-parameter Weibull",
            "weibull_shape_beta": round(rul.shape, 4),
            "weibull_scale_eta_hours": round(rul.scale, 2),
            "current_age_hours": round(rul.current_age_hours, 2),
            "reliability": round(rul.reliability, 4),
            "rul_median_hours": round(rul.rul_median_hours, 2),
            "rul_p10_hours": round(rul.rul_p10_hours, 2),
            "rul_p90_hours": round(rul.rul_p90_hours, 2),
            "rul_trend_hours": (
                round(rul.rul_trend_hours, 2)
                if np.isfinite(rul.rul_trend_hours) else None
            ),
        },
    }
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def plot_feature_trends(path: str, timestamps: np.ndarray,
                        feature_history: dict,
                        di_series: np.ndarray) -> None:
    """绘制特征趋势与退化指标图 (2x2 子图)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("TBM Cutter Wear - AE Feature Trends", fontsize=14)

    specs = [
        ("rms", "RMS", "RMS amplitude"),
        ("kurtosis", "Kurtosis", "Kurtosis"),
        ("spectral_centroid", "Spectral Centroid", "Frequency (Hz)"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, specs):
        ax.plot(timestamps, feature_history[key],
                color="tab:blue", lw=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    ax = axes.flat[3]
    ax.plot(timestamps, di_series, color="tab:red", lw=1.5,
            label="Degradation Index")
    for thr, stage in zip(STAGE_THRESHOLDS, STAGE_LABELS_EN[1:]):
        ax.axhline(thr, ls="--", lw=0.8, color="gray", alpha=0.7)
        ax.text(timestamps[0], thr + 0.01, stage, fontsize=8, color="gray")
    ax.axhline(1.0, ls="-", lw=1.0, color="k", alpha=0.8)
    ax.text(timestamps[0], 1.01, "Failure threshold", fontsize=8)
    ax.set_title("Degradation Index")
    ax.set_ylabel("DI")
    ax.set_ylim(0, max(1.2, float(np.max(di_series)) * 1.1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes.flat[2:]:
        ax.set_xlabel("Operating time (h)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
