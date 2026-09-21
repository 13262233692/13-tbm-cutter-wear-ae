"""磨损状态判别与威布尔剩余寿命（RUL）预测模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# 磨损状态分级阈值（基于综合退化指数 DI ∈ [0, 1]）
WEAR_STATES = [
    (0.30, "normal", "正常"),
    (0.65, "mild", "轻度磨损"),
    (1.01, "severe", "严重磨损"),
]


@dataclass
class FeatureBaseline:
    """新刀状态特征基线，用于归一化退化指数。"""
    rms: float
    kurtosis: float
    spectral_centroid: float


def degradation_index(features: dict, baseline: FeatureBaseline) -> float:
    """综合退化指数：RMS 上升、频谱质心下移、峰度变化加权融合，裁剪到 [0, 1]。"""
    rms_ratio = features["rms"] / max(baseline.rms, 1e-12)
    centroid_drop = 1.0 - features["spectral_centroid"] / max(baseline.spectral_centroid, 1e-12)
    kurt_ratio = features["kurtosis"] / max(baseline.kurtosis, 1e-12)

    di = (
        0.55 * min(max(rms_ratio - 1.0, 0.0) / 3.5, 1.0)
        + 0.25 * min(max(centroid_drop, 0.0) / 0.5, 1.0)
        + 0.20 * min(max(kurt_ratio - 1.0, 0.0) / 3.0, 1.0)
    )
    return float(np.clip(di, 0.0, 1.0))


def smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    """滑动平均平滑，抑制单窗口特征噪声，保持序列长度。"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    # 端点用原始值回填，避免边界低估
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def classify_wear(di: float) -> tuple[str, str]:
    for threshold, code, label in WEAR_STATES:
        if di < threshold:
            return code, label
    return "severe", "严重磨损"


def fit_weibull(times: np.ndarray, di_values: np.ndarray) -> dict:
    """对退化指数历史序列拟合威布尔退化模型。

    将 DI 视为累积损伤，拟合两参数威布尔 CDF：
        DI(t) = 1 - exp(-(t/eta)^beta)
    通过线性化 ln(-ln(1-F)) = beta*ln(t) - beta*ln(eta) 估计参数。
    """
    times = np.asarray(times, dtype=np.float64)
    di_values = np.asarray(di_values, dtype=np.float64)

    # 仅使用已进入退化区间且未饱和的样本
    mask = (di_values > 0.02) & (di_values < 0.98) & (times > 0)
    t_fit, f_fit = times[mask], di_values[mask]
    if len(t_fit) < 3:
        return {"beta": None, "eta": None, "n_samples": int(len(t_fit)),
                "note": "insufficient degradation samples"}

    x = np.log(t_fit)
    y = np.log(-np.log(1.0 - f_fit))
    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    beta = float(slope)
    eta = float(np.exp(-intercept / slope))
    return {"beta": beta, "eta": eta, "r_squared": float(r_value ** 2),
            "n_samples": int(len(t_fit))}


def predict_rul(weibull: dict, current_di: float,
                failure_threshold: float = 0.85) -> dict:
    """由威布尔模型预测到达失效阈值的剩余寿命（与 times 同单位）。

    以当前 DI 在模型上对应的等效时间为起点，保证 RUL 与当前退化水平一致。
    """
    beta, eta = weibull.get("beta"), weibull.get("eta")
    if not beta or not eta or beta <= 0:
        return {"rul": None, "failure_time": None,
                "note": "weibull model unavailable"}
    # 反解 CDF: t(F) = eta * (-ln(1-F))^(1/beta)
    failure_time = eta * (-np.log(1.0 - failure_threshold)) ** (1.0 / beta)
    di_clip = float(np.clip(current_di, 0.0, failure_threshold))
    current_equiv = eta * (-np.log(1.0 - di_clip)) ** (1.0 / beta)
    rul = max(failure_time - current_equiv, 0.0)
    return {"rul": float(rul), "failure_time": float(failure_time),
            "failure_threshold": failure_threshold}

def evaluate_sensor(feature_history: list[dict], times: np.ndarray,
                    baseline: FeatureBaseline,
                    failure_threshold: float = 0.85) -> dict:
    """单传感器完整评估：DI 序列、磨损分级、威布尔拟合与 RUL。"""
    di_raw = np.array([degradation_index(f, baseline) for f in feature_history])
    di_series = smooth_series(di_raw)
    code, label = classify_wear(di_series[-1])
    weibull = fit_weibull(times, di_series)
    rul = predict_rul(weibull, float(di_series[-1]), failure_threshold)
    return {
        "di_series": di_series,
        "current_di": float(di_series[-1]),
        "wear_state": code,
        "wear_state_label": label,
        "weibull": weibull,
        "rul": rul,
    }
