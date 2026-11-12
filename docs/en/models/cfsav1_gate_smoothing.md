# CFSAv1 调制系数平滑实验

此变体只平滑调制系数，三个频带分量、残差路径和损失保持原样：

```python
delta = rho * gate(condition).tanh()
delta = F.avg_pool2d(delta, 3, stride=1, padding=1, count_include_pad=False)
output = input + sum(delta_k * band_k)
```

`count_include_pad=False` 避免边缘系数被补零稀释。零初始化时输出仍等于输入；
没有新增可训练参数。训练、验证和推理均使用相同的平滑方式。

## 开关

在 `model.train()` 中传入 `cfsav1_gate_smooth=True`。默认 False 保持原版行为。
旧 CFSAv1 检查点没有这个属性时按 False 处理；新检查点保存开关。
不要仅在验证时对旧模型开启平滑，并将结果当成重新训练的平滑变体。

`seg_train.py` 的 `CFSAV1_GATE_SMOOTH` 控制开关，输出名称随开关区分。
本次为从头训练：rho=0.1、seed=42、1600、rect=True、batch=32、AMP、
workers=0、最多200轮、patience=30；配对和敏感性监督关闭。

## 如何判断效果

对照为已有 rho=0.1、不平滑的实验。统一沿用项目的 Box+Mask 综合 fitness
选择 best.pt，不修改早停指标。对照 best.pt 的 Mask Recall=0.57086、
mAP50=0.60980、mAP50:95=0.26242；其全程最高 Mask mAP50:95=0.26436。
无模块参照 best.pt 的 Mask mAP50:95=0.27885。

先比较统一 best.pt 指标，再辅助观察最高 Mask mAP 和曲线。
超过不平滑版只说明这次平滑有帮助，不等于超过基线或有稳定泛化收益。
候选改善后，需用相同的多个随机种子做配对重复，并报告波动。
平滑也可能缩小调制幅度，因此指标提高不能单独证明空间连续性是原因。

## 工程验证

`tests/test_cfsav1.py` 覆盖显式局部均值公式、边缘处理、恒等初始化、梯度、
旧模型兼容、保存加载、参数传递、合成数据训练/续训、CUDA AMP和追踪。
工程测试通过不代表真实数据集指标提高。
