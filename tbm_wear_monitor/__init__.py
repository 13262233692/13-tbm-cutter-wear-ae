"""TBM 刀盘刀具磨损声发射监测系统.

模块组成:
    ae_acquisition   -- 声发射信号采集与 HDF5 存储
    feature_extractor-- 小波降噪与特征提取 (RMS / 峰度 / 频谱质心)
    wear_model       -- 基于威布尔分布的剩余寿命预测
    report_generator -- 趋势可视化与 JSON 报告输出
"""

__version__ = "1.0.0"
