# TBM 刀盘刀具磨损声发射监测系统

基于声发射 (AE) 技术的盾构机刀盘刀具磨损监测与剩余寿命预测系统。

## 系统架构

```
8 通道 AE 传感器 (1 MHz)
        │
        ▼
 ae_acquisition    波形采集 / 仿真, HDF5 存储
        │
        ▼
 feature_extractor 小波阈值降噪 + RMS / 峰度 / 频谱质心
        │
        ▼
 wear_model        磨损分级 (DI) + 威布尔剩余寿命预测
        │
        ▼
 report_generator  JSON 报告 + 特征趋势图 (PNG)
```

## 模块说明

| 模块 | 功能 |
| --- | --- |
| `tbm_wear_monitor/ae_acquisition.py` | 8 通道 1 MHz 波形采集（内置物理一致的仿真器），gzip 压缩写入 HDF5 |
| `tbm_wear_monitor/feature_extractor.py` | db6 小波软阈值降噪 (VisuShrink)，提取 RMS、峰度、频谱质心 |
| `tbm_wear_monitor/wear_model.py` | 三特征融合退化指标 DI、四级磨损分级、两参数威布尔 RUL 预测 |
| `tbm_wear_monitor/report_generator.py` | 生成 JSON 监测报告与 2x2 特征趋势图 |
| `main.py` | 端到端监测流程入口 |

## 磨损分级与寿命预测

- 退化指标 DI ∈ [0, 1]：正常 < 0.25 ≤ 初期磨损 < 0.50 ≤ 中度磨损 < 0.80 ≤ 严重磨损，DI = 1.0 为失效阈值
- 威布尔模型：历史失效样本 MLE 拟合 (β, η)，输出当前可靠度 R(t) 与条件剩余寿命中位数及 P10/P90 置信区间
- 趋势外推：对近期 DI 序列做滑动平均 + 指数拟合，外推到达失效阈值的剩余时间

## 使用方法

```bash
pip install -r requirements.txt
python main.py --hours 120 --step 4 --output-dir output
```

输出文件（`output/`）：

- `ae_data.h5` — 原始波形（`/windows/<idx>/waveforms`，形状 8×50000）及采集元数据
- `wear_report.json` — 磨损状态、特征摘要、剩余寿命预测
- `feature_trends.png` — RMS / 峰度 / 频谱质心 / DI 趋势图

## 接入真实硬件

将 `AEAcquisition.acquire()` 替换为真实采集卡接口即可，下游特征提取、
寿命模型与报告模块无需改动；`BASELINE` 基准值需用新刀数据重新标定。
