"""盾构机刀盘磨损监测系统主流程。

流程：AE 采集(模拟) -> HDF5 存储 -> 小波降噪 + 特征提取
     -> 磨损分级 + 威布尔 RUL 预测 -> JSON 报告 + 趋势图。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ae_acquisition import AcquisitionConfig, acquire_window, save_hdf5, load_hdf5
from feature_extractor import extract_all_channels, wavelet_denoise
from wear_model import FeatureBaseline, evaluate_sensor
from report_generator import build_report

OUTPUT_DIR = Path("output")
DATA_DIR = OUTPUT_DIR / "hdf5"
FIG_DIR = OUTPUT_DIR / "figures"

NUM_EPOCHS = 40          # 监测历程采样点数
TOTAL_HOURS = 500.0      # 监测总时长（小时）
FAILURE_THRESHOLD = 0.85


def run_monitoring() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    times = np.linspace(1.0, TOTAL_HOURS, NUM_EPOCHS)
    base_time = datetime(2026, 9, 21, 8, 0, 0)

    # 1) 逐时刻采集并写入 HDF5，再读回做特征提取（模拟在线监测历程）
    feature_history: list[list[dict]] = []  # [epoch][sensor]
    for epoch, t in enumerate(times):
        wear_stage = t / TOTAL_HOURS
        config = AcquisitionConfig(wear_stage=wear_stage, seed=epoch)
        waveforms = acquire_window(config)
        h5_path = DATA_DIR / f"ae_epoch_{epoch:03d}.h5"
        save_hdf5(str(h5_path), waveforms, config,
                  timestamp=(base_time + timedelta(hours=float(t))).isoformat())

        loaded, attrs, layout = load_hdf5(str(h5_path))
        features = extract_all_channels(loaded, attrs["sampling_rate"], denoise=True)
        feature_history.append(features)

    feature_history = np.array(feature_history, dtype=object)  # (epoch, sensor)
    n_sensors = feature_history.shape[1]

    # 2) 以首个时刻（新刀）各传感器特征作为独立基线
    first = feature_history[0]
    baselines = [
        FeatureBaseline(rms=f["rms"], kurtosis=f["kurtosis"],
                        spectral_centroid=f["spectral_centroid"])
        for f in first
    ]

    # 3) 每个传感器：退化指数 -> 磨损分级 -> 威布尔 RUL
    sensor_results = [
        evaluate_sensor(list(feature_history[:, s]), times, baselines[s],
                        FAILURE_THRESHOLD)
        for s in range(n_sensors)
    ]

    # 4) 可视化与报告
    from report_generator import (
        plot_feature_trends, plot_di_and_weibull, plot_denoise_comparison,
        write_json_report,
    )

    per_sensor_history = [list(feature_history[:, s]) for s in range(n_sensors)]
    fig_trends = plot_feature_trends(times, per_sensor_history,
                                     str(FIG_DIR / "feature_trends.png"))

    worst_idx = int(np.argmax([r["current_di"] for r in sensor_results]))
    fig_di = plot_di_and_weibull(
        times, sensor_results[worst_idx]["di_series"],
        sensor_results[worst_idx]["weibull"], FAILURE_THRESHOLD,
        str(FIG_DIR / "di_weibull.png"), sensor_name=f"(S{worst_idx + 1})")

    # 末时刻原始/降噪对比（最差传感器通道）
    loaded, attrs, _ = load_hdf5(str(DATA_DIR / f"ae_epoch_{NUM_EPOCHS - 1:03d}.h5"))
    raw = loaded[worst_idx]
    denoised = wavelet_denoise(raw)
    fig_denoise = plot_denoise_comparison(
        raw, denoised, attrs["sampling_rate"],
        str(FIG_DIR / "denoise_comparison.png"), sensor_name=f"(S{worst_idx + 1})")

    figures = {"feature_trends": fig_trends, "di_weibull": fig_di,
               "denoise_comparison": fig_denoise}
    meta = {
        "system": "TBM Cutter Wear AE Monitoring",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sampling_rate_hz": int(attrs["sampling_rate"]),
        "num_sensors": n_sensors,
        "failure_threshold_di": FAILURE_THRESHOLD,
    }
    report = build_report(sensor_results, layout, times, figures, meta)
    report_path = write_json_report(report, str(OUTPUT_DIR / "wear_report.json"))
    print(f"报告已生成: {report_path}")
    print(f"总体状态: {report['overall']['wear_state_label']} "
          f"(最差传感器 {report['overall']['worst_sensor']}, "
          f"DI={report['overall']['worst_di']:.3f})")
    if report["overall"]["min_rul_hours"] is not None:
        print(f"最小剩余寿命: {report['overall']['min_rul_hours']:.1f} h")
    return report


if __name__ == "__main__":
    run_monitoring()
