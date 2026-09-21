"""刀具磨损状态评估与威布尔剩余寿命预测模块.

磨损状态分级:
    基于归一化综合磨损指标 DI (degradation index) 划分
    正常 / 初期磨损 / 中度磨损 / 严重磨损 四个等级.

剩余寿命 (RUL) 预测:
    1. 对历史退化指标序列做指数趋势外推, 估计到达失效阈值的剩余时间;
    2. 利用历史失效样本拟合两参数威布尔分布, 计算当前时刻的
       条件可靠度与剩余寿命置信区间.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

WEAR_STAGES = ["正常", "初期磨损", "中度磨损", "严重磨损"]
STAGE_THRESHOLDS = [0.25, 0.50, 0.80]  # DI 分级阈值


@dataclass
class WearAssessment:
    """单个时间点的磨损评估结果."""

    timestamp: float
    degradation_index: float
    stage: str
    stage_index: int


def classify_wear(di: float) -> tuple[str, int]:
    """根据退化指标划分磨损等级."""
    idx = int(np.searchsorted(STAGE_THRESHOLDS, di, side="right"))
    return WEAR_STAGES[idx], idx


def compute_degradation_index(rms: float, kurtosis: float,
                              spectral_centroid: float,
                              baseline: dict) -> float:
    """融合三特征得到 [0, 1+] 的归一化退化指标.

    baseline 为新刀状态下各特征的基准值与量程:
        {"rms": (base, span), "kurtosis": (base, span),
         "spectral_centroid": (base, span)}
    """
    parts = []
    for name, value in (("rms", rms), ("kurtosis", kurtosis),
                        ("spectral_centroid", spectral_centroid)):
        base, span = baseline[name]
        parts.append(max(0.0, (value - base) / span))
    return float(np.clip(np.mean(parts), 0.0, 1.5))


@dataclass
class WeibullRUL:
    """威布尔寿命预测结果."""

    shape: float               # 形状参数 beta
    scale: float               # 尺度参数 eta (小时)
    current_age_hours: float   # 当前已运行时间
    reliability: float         # 当前时刻可靠度 R(t)
    rul_median_hours: float    # 剩余寿命中位数
    rul_p10_hours: float       # 剩余寿命 10% 分位 (保守)
    rul_p90_hours: float       # 剩余寿命 90% 分位
    rul_trend_hours: float     # 退化趋势外推的剩余寿命


class WeibullLifeModel:
    """两参数威布尔寿命模型."""

    def __init__(self, failure_threshold: float = 1.0):
        self.failure_threshold = failure_threshold
        self.shape: float | None = None
        self.scale: float | None = None

    def fit(self, failure_times: np.ndarray) -> None:
        """用历史失效时间样本拟合威布尔分布 (MLE)."""
        failure_times = np.asarray(failure_times, dtype=float)
        if len(failure_times) < 3:
            raise ValueError("威布尔拟合至少需要 3 个失效样本")
        shape, _, scale = stats.weibull_min.fit(failure_times, floc=0)
        self.shape = float(shape)
        self.scale = float(scale)

    def reliability(self, t: float) -> float:
        """可靠度 R(t) = exp(-(t/eta)^beta)."""
        self._check_fitted()
        return float(np.exp(-((t / self.scale) ** self.shape)))

    def conditional_rul(self, current_age: float,
                        confidence: float = 0.5) -> float:
        """条件剩余寿命: 已运行 current_age 未失效条件下,
        再运行 confidence 分位对应的时间."""
        self._check_fitted()
        # 条件失效概率 F(t+dt | 存活到 t) = confidence
        # => R(t+dt) = (1-confidence) * R(t)
        target_reliability = (1.0 - confidence) * self.reliability(current_age)
        t_future = self.scale * (-np.log(target_reliability)) ** (1.0 / self.shape)
        return float(max(0.0, t_future - current_age))

    def trend_based_rul(self, timestamps: np.ndarray,
                        di_series: np.ndarray,
                        current_time: float) -> float:
        """对退化指标做指数趋势拟合, 外推到达失效阈值的剩余时间."""
        di_series = np.clip(np.asarray(di_series, dtype=float), 1e-6, None)
        timestamps = np.asarray(timestamps, dtype=float)
        # 滑动平均抑制测量噪声, 提高外推稳定性
        if len(di_series) >= 5:
            kernel = np.ones(5) / 5.0
            padded = np.pad(di_series, 2, mode="edge")
            di_series = np.convolve(padded, kernel, mode="valid")
        # 仅使用最近一段退化数据拟合, 反映当前劣化速率
        n_fit = max(6, len(di_series) // 2)
        di_series = di_series[-n_fit:]
        timestamps = timestamps[-n_fit:]
        # log(DI) 对时间线性拟合 => DI(t) = a * exp(b*t)
        slope, intercept = np.polyfit(timestamps, np.log(di_series), 1)
        if slope <= 1e-9:
            return float("inf")
        t_fail = (np.log(self.failure_threshold) - intercept) / slope
        return float(max(0.0, t_fail - current_time))

    def predict(self, current_age_hours: float,
                timestamps: np.ndarray,
                di_series: np.ndarray) -> WeibullRUL:
        """综合输出剩余寿命预测."""
        self._check_fitted()
        return WeibullRUL(
            shape=self.shape,
            scale=self.scale,
            current_age_hours=current_age_hours,
            reliability=self.reliability(current_age_hours),
            rul_median_hours=self.conditional_rul(current_age_hours, 0.5),
            rul_p10_hours=self.conditional_rul(current_age_hours, 0.1),
            rul_p90_hours=self.conditional_rul(current_age_hours, 0.9),
            rul_trend_hours=self.trend_based_rul(
                timestamps, di_series, current_age_hours),
        )

    def _check_fitted(self) -> None:
        if self.shape is None or self.scale is None:
            raise RuntimeError("模型尚未拟合, 请先调用 fit()")
