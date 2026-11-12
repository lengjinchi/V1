# CFSAv1 渐进调制实验

保留原 CFSAv1 的分频、门控、残差和零初始化，在调制系数上乘训练进度系数：

```python
alpha = start + (1 - start) * min(epoch / ramp_epochs, 1)
delta = alpha * rho * tanh(gate(condition))
output = input + sum(delta_k * band_k)
```

`epoch` 从 0 开始。设 `ramp_epochs=20`、`start=0.1`、`rho=0.25` 时：

| 当前训练轮次 | 已完成轮数 | alpha | 有效调制上限 |
|---|---:|---:|---:|
| 第 1 轮 | 0 | 0.1 | 0.025 |
| 第 11 轮 | 10 | 0.55 | 0.1375 |
| 第 21 轮及以后 | ≥20 | 1 | 0.25 |

## 设置

`model.train()` 接受两个新参数：

- `cfsav1_gate_ramp_epochs=20`：20 轮后使用完整调制；默认 0，关闭此策略。
- `cfsav1_gate_ramp_start=0.1`：起始比例，范围 `(0, 1]`。

已将 `seg_train.py` 配置为从头训练、不平滑、rho=0.25、前 20 轮渐进调制。
脚本中的 `CFSAV1_GATE_RAMP=False` 会把时长设为 0，恢复原版调制方式。
仍保留 1600、rect=True、batch=32、seed=42、200 轮和 patience=30。
实验目录名称包含 ramp 时长，避免与已有训练混淆。

该策略不冻结主干，不使用教师、参考权重或配对视图，也不改变学习率计划。
`cfsav1_warmup_epochs` 是原有的“暂时只更新增强模块”策略，含义不同，当前保持为 0。

## 验证、EMA 和恢复

`gate_scale` 是随检查点保存的 buffer，不是可学习参数。
训练每轮首个 batch 按绝对 epoch 更新学生和 EMA；每次优化后重新同步 EMA 的比例，
避免 EMA 对比例做移动平均。前 20 轮验证使用当轮比例；后续使用 1。
独立验证与推理使用检查点中保存的比例，不强行把早期检查点切换为 1。
恢复训练从恢复后的 epoch 计算比例，不从 0.1 重新开始。

旧 CFSAv1 整体模型文件缺少该 buffer 时按比例 1 推理；旧 state_dict 加载缺失的
buffer 时补为 1。原 CFSA 不变。

## 实验解释

这是一个尚未证实有效的训练策略变体。零初始化只保证训练开始时恒等；渐进系数
控制最初 20 轮更新后调制的最大强度。工程检查通过不等于真实数据集性能提升。
对比已有不平滑、rho=0.25 实验，沿用相同数据、seed、训练预算和最佳权重选择规则。
重点报告总体 Mask mAP、麻点/划痕类别 AP、统一阈值 Recall 和 Box 指标。
