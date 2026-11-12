# SAMX 三项后续实验

本轮新增实现位于 `vajra/nn/samx_followups.py`。原版 SAMX、StageGlobal、LocalGlobal 和共享路径几何残差均保留，旧 `seg_train.py` 默认不变。已有实验目录和权重不作删除或覆盖。

## 模块与开关

| 实验 | P4 samx_mode | samx_proto_mode | 对照 |
| --- | --- | --- | --- |
| dual_geometry | geometry_dual | off | geometry_residual，共享路径版 |
| proto_free | local_global | free | local_global，原型增强关闭 |
| proto_structured | local_global | structured | proto_free |

其他新参数：`samx_proto_channels=16`、`samx_proto_gain=0.1`。偏移和形状参数沿用 `samx_offset_lr_scale=0.1`。这些开关独立于 CFSA/CFSAv1；Proto 几何增强不能与 CFSA/CFSAv1 同时开启。

### 双路径几何残差

每阶段普通路径和几何路径使用各自的特征提取参数，初始从同一组权重复制。固定采样和偏移采样仍共享几何路径的特征与线性聚合权重；差值中不增加 BN 或激活。共同权重和初始输出与共享路径版一致，后续训练两套特征提取参数可分别学习。独立路径同时增加参数量，结果不能解释为等容量实验。

### P3 原型生成前的自由采样

`P3 → 1×1 压缩至16通道 → 三点几何差值 → 1×1 回投影 → 乘可学习系数 → 加回P3 → Proto`。

该模块只在原型生成路径中调用，固定权重前向时不会改动检测和掩膜系数的输入；训练中的共享参数梯度仍可能间接影响其他分支。P4 保持 LocalGlobal 结构。偏移零初始化保证初始原型输入恒等，输出投影正常初始化、系数从非零 0.1 开始，偏移分支可收到任务梯度。

### 成组旋转伸缩采样

保持第二项的投影、通道、三点模板、聚合层和 P4 设置。只把各点独立的六个偏移量，改为局部旋转角及两个方向的缩放因子，共三个预测场。以三点模板重心为中心旋转、伸缩，不预测整体平移。

角度 `π*tanh(t)`、两方向缩放 `exp(log(2)*tanh(s))`；初始角度 0、缩放 1。代码以相对原模板的偏移表达变换，确保初始修正严格为零，避免重心浮点运算产生微小非零偏移。坐标与变换计算保持 FP32，采样特征保持 AMP/模型的数据类型。

## 执行入口

固定配置在 `samx_followup_config.py`。完整计划和保留清单位于 `output/samx_followups_20261029/执行计划.md`。

```powershell
# 先运行检查后的三项真实数据试跑和旧模型复现
D:/anaconda3/envs/vajrav1/python.exe -u run_samx_followups.py --preflight

# 预检全部通过后，串行训练三项，每项结束自动验证、再启动下一项
D:/anaconda3/envs/vajrav1/python.exe -u run_samx_followups.py --queue
```

脚本检查源文件快照，拒绝重复启动、覆盖或静默改名。预检和正式队列分别有状态文件；子进程失败时停止队列并保留错误。这里不排队已取消的修复原版对照。

## 输出与评价

- `status.json`：正式队列、当前子进程及其最新进度。
- `*_progress.json` / `*.log`：每项进度与日志。
- `*_result.json`：最佳权重的完整 FP32 验证、逐类指标、边界和速度结果。
- `baseline_*_result.json`：旧 LocalGlobal 与共享几何版的完整复现及补充评价。
- `comparison.json` / `结果汇总.md`：按预定比较对象汇总结果。
- `protected_existing_weights.json`：本轮开始前 28 个旧权重的保护清单，旧 best 另核对 SHA256。

边界指标是像素微平均 F1：固定 conf=0.25，以同类 Mask IoU≥0.5 做一对一贪心匹配，容差为 1 个掩膜像素的方形邻域，本设置对应约 4 个输入像素。未匹配预测和 GT 也进入边界精确率/召回率的分母。它是本轮补充指标，与 AP 分开报告。

速度采用 FP32、batch1、288×1632 网络输入，20 次预热、50 次计时，排除数据加载、NMS 和掩膜重建；同时记录增量峰值已分配显存。该 FPS 不与论文中的不同推理配置混比。
