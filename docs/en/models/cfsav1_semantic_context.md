# CFSAv1 P4 空间语义提示实验

前两项诊断发现，当前最佳权重的频带增益随空间位置变化很小，去掉这种变化后分割指标也几乎不变。
这次实验只检验一个假设：引入更深层 P4 的空间信息，是否有助于门控选择不同区域的频带证据。

P4 提示不是缺陷真值图，也没有额外的边界预测损失；它通过已有分割任务学习。
本实验不预先保证门控空间差异一定变大或指标一定提高。

## 结构

P3 继续用于低、中、高三个分量的分解，局部和全局门控保留。
新增 P4 分支：1×1卷积压缩至16通道 → SiLU → 深度3×3卷积 → SiLU → 1×1卷积输出12个logits。
先在 P4 的低分辨率网格计算，再将12通道提示双线性插值到 P3 大小，align_corners=False。
这个操作是特征网格的尺度匹配，不宣称恢复了严格的像素级感受野对齐。

```text
z_local  = local_gate(P3 context, grouped band responses)
z_global = global_gate(global average of the original condition)
z_p4     = resize(P4_gate(P4), P3 size)
delta    = rho * tanh(z_local + z_global + z_p4) * gate_scale
P3_out   = P3 + sum(delta_k * band_k)
Proto    = original_Proto(P3_out)
```

新增分支末层零初始化，训练开始时仍然精确恒等。梯度先更新末层，随后传入较早的投影层和 P4。
三个分支合并后只经过一次 tanh，因此每个分组、每个频带的增益范围仍为[0.75,1.25]。
检测分支和掩膜系数分支的直接输入不变；联合训练时，新增梯度仍可能影响共享主干和这些分支。
没有加入归一化、额外配对、教师模型、参考权重或额外损失。

## 开关与兼容

```python
CFSAV1_SEMANTIC_CONTEXT = True
cfsav1_semantic_context=CFSAV1_SEMANTIC_CONTEXT
```

框架默认 False。开启时要求 use_cfsav1=True；关闭后恢复局部＋全局版本。
旧完整模型检查点缺少 semantic_gate 时继续走原路径。新增开关和参数随新检查点保存。
不将没有训练过的 P4 分支当作已有改进模型直接评价；正式实验从模型名称建立新模型训练。

## 本次对照

- 对照：已完成的局部＋全局实验，最佳轮次117；本次不覆盖其目录或权重。
- 新实验：局部＋全局＋P4提示。
- 数据划分、200轮上限、patience=30、img_size=1600、rect=True、batch=32、AMP、seed=42等设置保持一致。
- rho=0.25，gate_smooth=False，gate_ramp_epochs=0，warmup_epochs=0；从头训练。
- 新name：VajraV1_CFSAV1_rho0.25_global1_semanticP4_smooth0_ramp0_start1.0_seed42。
- 本次为单种子探索性对照，不能据此确定跨训练随机性的稳定收益。

## 完成后如何判断

首先用相同独立验证设置重测两个最佳权重，比较 Mask mAP50:95、mAP50、P/R 和各类别 AP。
随后对新权重重复空间均值干预和区域权重统计，给它们设置独立输出目录。
若空间差异增大但指标下降，不能将门控变化大解释为创新机制有效。
若指标提升但空间均值干预仍无影响，也不能将提升直接归因于按位置选择频带。

```powershell
python diagnose_cfsav1_spatial.py --weights '新实验/weights/best-vajra-v1-nano-seg.pt' --output output/cfsav1_semantic_spatial_diagnostic
python diagnose_cfsav1_regions.py --weights '新实验/weights/best-vajra-v1-nano-seg.pt' --output output/cfsav1_semantic_region_diagnostic
```

独立诊断已经包含 P4 logits 的缩放和相加，统计的是实际合并后的门控。
代码修改前备份位于 output/cfsav1_semantic_p4_20261021/before_semantic.zip。
