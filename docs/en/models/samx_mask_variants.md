# SAMX 原型/系数协同与有界偏移实验

本轮保留所有原模块和已有权重，新增两个默认关闭的开关。实现位于 `vajra/nn/samx_mask_variants.py`。原 `head.py`、SAMX、几何差值、CFSA/CFSAv1 及原 `seg_train.py` 均不修改；只增加模型/配置/训练器的参数传递。

## 开关

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `samx_proto_share_coeff` | `False` | 为 True 时，增强后的 P3 同时供给 Proto 和 P3 掩膜系数分支。只共享输入特征，不共享这两条分支的参数。 |
| `samx_proto_offset_bound` | `0.0` | 为 0 时保留原自由偏移；为正数 r 时，Proto 分支的每轴偏移变为 `r*tanh(o/r)`，单位是 P3 特征格。 |

两个新开关均要求 `samx_proto_mode="free"`，与 CFSA/CFSAv1 互斥。它们不改变 P4 LocalGlobal 中已有 LDConv 的偏移。默认值恢复旧 Proto 自由采样行为，旧完整对象权重仍可直接加载。

## 独立对照

1. `proto_coeff_shared`：仅打开共享输入。增强只计算一次，Proto 与 P3 系数分支消费同一个增强结果；检测和 P4/P5 系数分支仍消费原始输入。
2. `proto_offset_bounded`：只启用有界 Proto 偏移，系数分支接入保持原样。上限依据旧 proto_free 在固定 128 张训练图像上的绝对偏移坐标 p95，选择不小于 `max(1,p95)` 的最小 2 的幂，正式训练前冻结。

两项均独立从头训练，直接对照已有 proto_free，不把两种改动叠加，也不加载旧权重初始化学生模型。统计旧模型只用于选定第二项的位移上限。

上限单位是“每轴坐标”，不是欧氏半径；P3 步长为 8，r 对应输入每轴最多 8r 像素的附加位移。该 tanh 变换会改变小于 r 的偏移，不能解释成只截断极端值。其零点导数为 1，避免仅因 r 不同而缩放初始偏移梯度。

## 固定设置

两项保持相同：200 轮上限、patience=30、img_size=1600、rect=True、batch=32、workers=0、AMP、seed=42、SGD lr0=0.01、momentum=0.9、warmup_bias_lr=0。偏移参数维持 0.1 倍学习率，包含预热阶段；Proto 压缩通道 16，残差可学习系数初始 0.1。CFSA/CFSAv1/MABD、配对、参考监督、冻结和渐进门控均不参与。

## 检查与队列

配置：`samx_mask_experiment_config.py`。

执行入口：`run_samx_mask_experiments.py --queue`。已有队列运行时不要重复启动。

先检查旧实验已正常结束及 34 个旧检查点，再依次执行：固定训练图偏移诊断、旧 proto_free 完整 FP32 复现、两项各 1% 数据单轮试跑与完整验证、正式协同版训练及独立验证、正式有界版训练及独立验证。不同阶段使用独立子进程，失败停止并保存日志，不静默修改 batch 或尺寸。

结果保存在 `output/samx_mask_experiments_20261030/`。`status.json` 记录阶段与轮次；`offset_diagnostic.json` 固定上限与抽样图像；`*_result.json` 保存最佳权重完整验证；`结果汇总.md` 保存正式结果。只用完整训练后的独立验证判断效果，不把单轮试跑当作精度结论。

继续沿用此前的最佳权重选择、FP32 验证、逐类指标、边界 F1 和纯网络前向速度协议。主看 Recall、AP50 能否恢复，同时保持 AP50:95 和边界指标，速度代价也需报告。

有界偏移和几何对齐已有相关研究，本轮定位为性能消融，尚无收益或原创性承诺。
