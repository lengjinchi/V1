# PPT 版 CFSAv1 + SAMX 联合实验

依据：`C:/Users/Administrator/Desktop/11.8/zjx_方法汇报精简版_20261108.pptx`。

## 模块设置

- CFSAv1：P3 → Proto 分支，局部分频，低/中/高三个分量；4 组、上下文宽度 16，rho=0.25。
- CFSAv1 全局上下文、P4 语义上下文、权重平滑、20 轮渐进门控、冻结训练全部关闭。
- SAMX：仅 nano neck3/P4，`local_global`，局部得分 + 全局得分后 softmax 融合两路特征。
- LDConv 偏移参数学习率是主学习率的 0.1，包含常规优化器 warmup。
- 原 CFSA、MABD、P3 几何残差模块关闭。配对训练和参考监督不启用。

## 训练设置

从架构名新建模型，不读取已有权重、不 resume。数据沿用原实验配置（四类钢材表面缺陷；训练 6134 张、验证 1532 张），输入 1600，rect=True，batch=32，AMP，workers=0，seed=42，SGD，lr=0.01，momentum=0.9，最多 200 轮，patience=30。

正式名称：`Joint_CFSAv1_local_rho0.25_SAMX_localGlobal_P4_img1600_seed42_20261109`。

## 执行清单

- [x] 保存配置、源码及 PPT 快照，使用独立目录，保留旧实验。
- [x] 10% 真实训练数据试跑 1 轮；等待 AMP 初始缩放适应后验证两模块实际参数更新、位置、优化器和有限损失。
- [x] 加载试跑权重并完整 FP32 验证，检查 1532 张 / 3942 实例。
- [x] 试跑通过后独立进程从头启动正式训练，进度和错误写入 JSON、日志。
- [x] 正式训练结束后，加载最佳权重，batch=8 完整 FP32 验证，保存指标与权重哈希。

运行命令：`D:/anaconda3/envs/vajrav1/python.exe -u run_joint_experiment.py --queue`。

进度目录：`output/joint_cfsav1_samx_20261109`，查看 `status.json`、`train_progress.json`、`train.log`。

最终结果：`train_result.json`。smoke 结果只用于检查流程，不作为正式实验指标。PPT 的手工数值不作为对照；最终比较应使用同一验证口径。历史 CFSAv1 global1 结果不能当作本次 local-only 的匹配单模块对照。

## 启动记录

2026-11-09 10:07:09 正式 worker 启动，PID 8068；队列 PID 9076。
已检查实际加载 6134 张训练图，每轮 192 个 batch；两个模块及偏移学习率分组正确。
试跑最终权重 SHA256：`a2211033d4929783468b8a2ce4aab750b5a1c1f89f31fc43a1f438eed8abe7b1`。
初始短试跑和检查脚本修正的记录保留在审计目录的 `attempt*` 文件夹，均未进入正式训练。

## 完成与复核

正式实验于 2026-11-09 16:02 完成：第 146 轮触发早停，最佳权重来自第 116 轮。
独立 FP32 验证 Mask P=69.0151%、R=54.1974%、AP50=61.8120%、AP50:95=26.8358%。
权重 SHA256：`70384b121c907bf14e349879316a57f4e1d5a68fbb69cd4319b5e24e8bc08fad`。

同日补做已有 Baseline、局部 CFSAv1 权重的相同 FP32 验证；四项对比见审计目录的 `comparison.md` 与 `comparison_verified.json`。
联合模型较 SAMX 的 AP50/AP50:95 增加 0.5024/0.4279 个百分点，但 Recall 下降 6.0687 个百分点；相对已有 Baseline 权重，两项 AP 仍低 0.5595/0.4056 个百分点。
历史 Baseline/CFSAv1 训练设置与本次不完全一致，这些复测用于性能参照；不能将差值全部归因于模块。所有现有实验和权重均保留。
