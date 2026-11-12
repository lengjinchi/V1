# MABDv1：恢复原 Dice、保留 thinness 边界权重（2026-11-10）

本轮新增独立模式 `mask_loss_mode='mabd_v1_boundary'`。所有已有模式、权重、实验输出均保留。

上一轮 `mabd_v1_weights` 同时用 thinness 控制 Dice 和边界权重，在原 GT 框内监督，召回率增加但 AP50、AP50:95 下降。本轮只将 Dice 权重恢复为原 MABD 的框长宽比权重，检查两项同时使用 thinness 的影响。该解释尚未被实验证实。

```
shape = clip(log(max(w / h, h / w)) / log(50), 0, 1)
q = sum(G - erode_3x3_zero_padded(G)) / max(sum(G), eps)
dice_weight = 0.30 + 0.20 * shape          # 调用原 morphology_factors
boundary_weight = 0.05 + 0.15 * q         # 沿用上一轮，范围 0.05~0.20
loss = BCE_GT_box + dice_weight * Dice_GT_box + boundary_weight * BoundaryBCE_GT_box
```

本轮与上一轮 `thinness + 原 GT ROI` 仅改变 Dice 权重来源；边界带、原 GT 框、归一化、FP32 归约、正样本平均均不变。与历史原 MABD 比较时，须注明历史实现的数值精度及图像边缘腐蚀处理不同，不能声称相对历史代码只有一行权重变化。

执行清单：

1. 单独添加新损失模式；直接复用原 Dice 形态因子的实现。
2. 检查平方/细长框的 Dice 权重、与上一轮的单项差异、框外零梯度、内部掩膜相对原损失的解析差值、配置及保存加载和 AMP。
3. 核对原 MABD、完整版、上一轮仅权重版的权重哈希、数据 YAML 和验证协议。
4. 10% 真实数据训练 1 轮，核验实际参数更新、保存加载及 1532 张全验证集 FP32/边界评估。
5. 自动从头训练唯一新变体：300 轮上限、patience=30、1600、rect=True、batch=32、AMP、seed=42、SGD(.01,.9)。
6. CFSA、CFSAv1、SAMX、Proto 几何变体关闭；不加载旧权重初始化。
7. 最佳权重自动统一评估，生成四组对比；单种子只用于初步判断趋势。

入口：`D:\anaconda3\envs\vajrav1\python.exe -u run_mabd_boundary_experiment.py --queue`。

实验名：`MABDv1_origDice_thinBoundary_gtROI_base_img1600_seed42_20261110`。

输出：`output/mabd_v1_boundary_20261110/`；训练权重：`MABD compare/<实验名>/weights/`。
