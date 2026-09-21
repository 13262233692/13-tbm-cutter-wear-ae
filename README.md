# 盾构机刀盘磨损声发射监测系统

基于 8 通道声发射（AE）传感器（1 MHz 采样）的刀盘刀具磨损监测与剩余寿命预测系统。

## 模块结构

| 模块 | 职责 |
|---|---|
| `ae_acquisition.py` | 8 通道 AE 波形采集（含仿真信号源），HDF5 存取 |
| `feature_extractor.py` | 小波阈值降噪，提取 RMS / 峰度 / 频谱质心 |
| `wear_model.py` | 综合退化指数、磨损三级分类、威布尔 RUL 预测 |
| `report_generator.py` | JSON 报告生成与特征趋势可视化 |
| `main.py` | 端到端监测流水线 |

## 运行

```bash
pip install -r requirements.txt
python main.py
```

## 输出

- `output/hdf5/ae_epoch_*.h5` — 各监测时刻的原始波形与元数据
- `output/wear_report.json` — 各传感器磨损状态、威布尔参数与剩余寿命
- `output/figures/feature_trends.png` — RMS / 峰度 / 频谱质心趋势
- `output/figures/di_weibull.png` — 退化指数与威布尔拟合曲线
- `output/figures/denoise_comparison.png` — 小波降噪前后波形/频谱对比

## 方法说明

- **降噪**：db6 小波 5 层分解 + VisuShrink 通用阈值软阈值处理。
- **退化指数 DI**：RMS 上升、频谱质心下移、峰度变化加权融合，归一化到 [0,1]。
- **磨损分级**：DI < 0.30 正常；0.30–0.65 轻度磨损；≥ 0.65 严重磨损。
- **寿命预测**：对 DI 历史序列线性化拟合两参数威布尔分布，反解到达失效阈值（DI=0.85）的时间。
