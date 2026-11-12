# CFSAv1 使用说明

原 CFSA（`FFTGuidedEMA`）保留，CFSAv1 位于 `vajra/nn/cfsav1.py`。
两者均作用于 Proto 输入，通过独立开关选择。

| 设置 | 效果 |
| --- | --- |
| `use_cfsa=False, use_cfsav1=False` | 基线 |
| `use_cfsa=True, use_cfsav1=False` | 原 CFSA |
| `use_cfsa=False, use_cfsav1=True` | CFSAv1 |

两个增强模块不能同时开启。当前训练只使用单视图分割监督；配对视图、参考模型校准、
敏感性辅助监督及相关训练参数已经移除。

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-seg")
model.train(
    data="your_data.yaml",
    use_cfsav1=True,
    use_cfsa=False,
    cfsav1_rho=0.25,
    cfsav1_global_context=True,
    cfsav1_gate_smooth=False,
    cfsav1_gate_ramp_epochs=0,
)
```

## 模块计算

- 固定高斯卷积构造 `L=G5(F)`、`M=G3(F)-G5(F)`、`H=F-G3(F)`。
- 不下采样，三个分量相加等于输入；它们是重叠宽频带，而非理想互斥频带。
- 默认 4 个通道组，16 个上下文通道。
- 根据局部上下文与分量能量生成局部门控。
- 开启全局上下文时，对同一门控输入做全局平均池化并映射，与局部门控 logits 相加。
- `delta=rho*tanh(logits)*gate_scale`，`F'=F+sum(delta_k*F_k)`。
- 末层零初始化，初始输出严格等于输入。

## 保留的可选功能

- `cfsav1_global_context`：全局上下文，默认 False。
- `cfsav1_gate_smooth`：仅平滑调制系数，默认 False。
- `cfsav1_gate_ramp_epochs`：渐进门控，默认 0 表示关闭。
- `cfsav1_warmup_epochs`：专用主干冻结预热，默认 0；与优化器 warm-up 不同。

详细说明见 `cfsav1_global_context.md`、`cfsav1_gate_smoothing.md`、`cfsav1_gate_ramp.md`。

## 历史权重兼容

原局部版、平滑版、渐进版和局部＋全局版的结构及已有权重保留。
旧配置中已关闭的配对/参考监督参数会在读取时移除，不再写入新训练配置。
尝试启用或恢复旧配对/参考监督训练会明确报错，不会静默改成另一种训练方式。

历史完整模型若含已训练的敏感性头，仅保留必要的推理兼容路径；新模型不会创建该头，
不再计算辅助损失，也不加载参考模型。历史含辅助损失的 CSV 仍能绘制常规指标。

本次清理前的源文件备份位于 `output/cfsav1_cleanup_20261019/before_cleanup.zip`。
实验目录、数据集、已训练权重不在清理范围内。

## 检查

```powershell
& 'D:/anaconda3/envs/vajrav1/python.exe' -m unittest discover -s tests -p 'test_cfsav1*.py'
```

覆盖恒等初始化、梯度、全局上下文、门控平滑、渐进门控、常规分割损失、AMP、
保存加载和合成数据训练/恢复。这些检查验证实现正确性，不代表分割精度提高。
