"""TBM 刀盘刀具磨损监测系统主流程.

流程:
    1. 采集 8 通道声发射波形并写入 HDF5
    2. 小波降噪 + 特征提取 (RMS / 峰度 / 频谱质心)
    3. 磨损状态分级 + 威布尔剩余寿命预测
    4. 输出 JSON 报告与特征趋势图

用法:
    python main.py [--hours 120] [--step 2] [--output-dir output]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tbm_wear_monitor.ae_acquisition import (
    AEAcquisition,
    AcquisitionConfig,
    load_window,
    open_acquisition_file,
)
from tbm_wear_monitor.feature_extractor import FeatureExtractor
from tbm_wear_monitor.report_generator import (
    generate_json_report,
    plot_feature_trends,
)
from tbm_wear_monitor.wear_model import (
    WeibullLifeModel,
    WearAssessment,
    classify_wear,
    compute_degradation_index,
)

# 新刀基准特征 (可通过标定更新): (基准值, 量程)
BASELINE = {
    "rms": (0.01, 0.70),
    "kurtosis": (180.0, 320.0),
    "spectral_centroid": (126_000.0, 80_000.0),
}


def wear_ratio_at(hour: float, total_hours: float) -> float:
    """演示用磨损演化曲线: 缓慢磨损后加速劣化."""
    x = hour / total_hours
    return float(np.clip(0.7 * x + 0.5 * x ** 3, 0.0, 1.2))


def run(total_hours: float, step_hours: float, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / "ae_data.h5"
    json_path = out / "wear_report.json"
    fig_path = out / "feature_trends.png"

    config = AcquisitionConfig()
    acquisition = AEAcquisition(config)
    extractor = FeatureExtractor(config.sampling_rate)

    timestamps = np.arange(0.0, total_hours + 1e-9, step_hours)
    feature_history = {"rms": [], "kurtosis": [], "spectral_centroid": []}
    di_series = []
    assessments: list[WearAssessment] = []

    print(f"[1/4] 采集声发射波形 -> {h5_path}")
    with open_acquisition_file(str(h5_path), config) as h5f:
        for idx, hour in enumerate(timestamps):
            wear_ratio = wear_ratio_at(hour, total_hours)
            waveforms = acquisition.acquire(wear_ratio)
            acquisition.save_window(h5f, idx, float(hour), wear_ratio,
                                    waveforms)

        print("[2/4] 小波降噪与特征提取")
        for idx, hour in enumerate(timestamps):
            waveforms, _ = load_window(h5f, idx)
            feats = extractor.process_window(waveforms, idx, float(hour))
            rms = feats.mean("rms")
            kurt = feats.mean("kurtosis")
            centroid = feats.mean("spectral_centroid")
            feature_history["rms"].append(rms)
            feature_history["kurtosis"].append(kurt)
            feature_history["spectral_centroid"].append(centroid)
            di = compute_degradation_index(rms, kurt, centroid, BASELINE)
            di_series.append(di)
            stage, stage_idx = classify_wear(di)
            assessments.append(WearAssessment(
                timestamp=float(hour),
                degradation_index=di,
                stage=stage,
                stage_index=stage_idx,
            ))

    print("[3/4] 威布尔寿命预测")
    # 历史失效样本 (同型刀具全生命周期统计, 小时)
    failure_samples = np.array([98, 105, 112, 118, 121, 127, 130, 136,
                                142, 148, 110, 125], dtype=float)
    model = WeibullLifeModel(failure_threshold=1.0)
    model.fit(failure_samples)
    rul = model.predict(
        current_age_hours=float(timestamps[-1]),
        timestamps=timestamps,
        di_series=np.asarray(di_series),
    )

    print("[4/4] 生成报告与趋势图")
    report = generate_json_report(
        str(json_path), assessments, rul, feature_history,
        num_sensors=config.num_sensors,
    )
    plot_feature_trends(str(fig_path), timestamps, feature_history,
                        np.asarray(di_series))

    print(f"  JSON 报告: {json_path}")
    print(f"  趋势图:    {fig_path}")
    print(f"  HDF5 数据: {h5_path}")
    print(f"  当前磨损状态: {report['current_status']['wear_stage']} "
          f"(DI={report['current_status']['degradation_index']})")
    print(f"  剩余寿命中位数: {rul.rul_median_hours:.1f} h "
          f"[P10={rul.rul_p10_hours:.1f}, P90={rul.rul_p90_hours:.1f}]")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TBM 刀盘磨损监测系统")
    parser.add_argument("--hours", type=float, default=120.0,
                        help="监测总时长 (小时)")
    parser.add_argument("--step", type=float, default=2.0,
                        help="采集间隔 (小时)")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    args = parser.parse_args()
    run(args.hours, args.step, args.output_dir)


if __name__ == "__main__":
    main()
