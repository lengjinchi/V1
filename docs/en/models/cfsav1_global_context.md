# CFSAv1 局部与全局上下文门控

通过 `cfsav1_global_context=True` 启用；框架默认 False，保持原 CFSAv1 行为。
`seg_train.py` 中的 `CFSAV1_GLOBAL_CONTEXT` 已开启，训练名称添加 `_global1`。
关闭后名称为 `_global0`，可作同条件对照。

## 计算方式

保留原低、中、高三个分量，低频也继续参与调制。原门控输入 `Z` 由局部上下文、
三分量能量组成。

```text
local_logits  = local_MLP(Z)                   # B, 3*groups, H, W
global_logits = global_MLP(mean_hw(Z))         # B, 3*groups, 1, 1
delta = rho * tanh(local_logits + global_logits) * gate_scale
output = input + sum(delta_k * band_k)
```

全局分支为全局平均池化 → 1×1卷积 → SiLU → 1×1卷积。
其输出按图像独立生成，在空间上广播；局部分支仍保留空间差异。
不混合不同样本的统计，不直接平滑分量，也不做固定低频或小波替换。
如开启原有门控平滑，则在合并后的 `delta` 上执行原有平滑操作。

两个分支末层均零初始化，初始输出严格等于输入。合并后只经过一次 tanh，
因此调制上限仍由 rho 和 gate_scale 决定，不因增加分支而翻倍。
默认 groups=4、context_channels=16 时新增 668 个可训练参数。
全局 MLP 在 1×1 空间上计算；新增池化和广播相加仍有开销，实际速度需实测。

## 当前实验设置

当前脚本为 rho=0.25、smooth=False、ramp=0（第一轮起完整调制强度）、batch=32、seed=42。
配对训练与参考监督已移除；模型只使用单视图分割监督。

比较时保持以上设置相同，仅切换 GLOBAL_CONTEXT。当前 batch=32，与最初结构版一致。

旧整模型检查点缺少全局分支时继续使用原路径；默认关闭时 state_dict 不增加参数键。
全局版检查点会保存全局分支参数和训练开关，恢复训练及推理保留该结构。
本次借鉴局部与全局联合生成权重的思路，不是 WPFormer 完整 WCA 的复现。
