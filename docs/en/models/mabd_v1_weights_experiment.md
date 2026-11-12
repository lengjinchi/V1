# MABDv1：只改形态权重的单项消融（2026-11-10）

本轮只训练 `mask_loss_mode='mabd_v1_weights'`，原 MABD、完整版及已有权重全部保留。

与完整版相比，仅将边界损失的监督区域从扩张框改回原 GT 框。仍用 3×3 形态学操作生成边界带；`mabd_v1_boundary_radius=1`，不能设为 0 代替关闭扩张。

两版都使用：

```
thinness = sum(G - erode_3x3(G)) / max(sum(G), eps)
dice_weight = 0.30 + 0.20 * thinness
boundary_weight = 0.05 + 0.15 * thinness
loss = BCE_GT_box + dice_weight * Dice_GT_box + boundary_weight * BoundaryBCE
```

本轮 BoundaryBCE 在原 GT 框内归一化，框外不产生该实例的掩膜损失梯度。它保留了上一轮的新权重定义，不等同于只替换原 MABD 的框几何量而完全保持旧权重公式。

## 执行清单

1. 保留原 BCE、MABD、完整 MABDv1 分支，添加独立权重模式。
2. 检查原框外梯度为零、框内监督存在、两版 thinness 与权重相同，以及默认完整版不变。
3. 核对历史两组权重哈希、数据 YAML 和验证协议，复用已有验证结果。
4. 在 10% 训练数据上跑 1 轮 smoke，检查有限损失、实际参数更新、存盘与全验证集独立评估。
5. smoke 通过后自动从头训练唯一新变体：最多 300 轮、patience=30、1600、rect=True、batch=32、AMP、seed=42。
6. 关闭 CFSA、CFSAv1、SAMX 和 Proto 几何变体；其他训练配置与上一轮完整版一致。
7. 对最佳权重执行同一 FP32 验证及边界 F1 评估，自动生成三组对比。

关注 Recall、Mask AP50、Mask AP50:95 和边界 F1 是否一起改善。上一轮 AP50:95 仅小幅增加而 Recall 下降；“扩张 ROI 导致模型更保守”只是待检验的解释，单次实验不能证明因果。

运行入口：`D:\anaconda3\envs\vajrav1\python.exe -u run_mabd_weights_experiment.py --queue`

独立实验名：`MABDv1_weightsOnly_gtROI_base_img1600_seed42_20261110`。
状态、日志、源码快照、对比结果写入 `output/mabd_v1_weights_20261110/`。
