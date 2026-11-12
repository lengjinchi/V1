# SAMX repaired sampling and separate routing experiments

## Modes

`samx_mode` supports `off` (default), `original`, `stage_global`, and `local_global`.
All enabled modes replace only `VajraV1ModelN.vajra_neck3` (192 -> 128 channels).
Unsupported architectures fail explicitly. Original `ShapeAdaptiveMerudandaX` remains
in `vajra/nn/modules.py`; separate variants live in `vajra/nn/samx.py`.

- `original`: repaired LDConv, original image-level two-way gate shared across stages.
- `stage_global`: two independent GAP gates, reading stage 2 and stage 3 inputs respectively.
- `local_global`: original global logits plus local 1x1 -> SiLU -> depthwise 3x3 ->
  SiLU -> 1x1 logits; softmax over two branches. The same spatial gate serves both
  stages. Local final-layer zero initialization recovers original routing initially.

Experiments 2 and 3 are separate ablations. Experiment 3 does not include independent
stage gates. Stage-global changes both gate independence and gate input source.
Both retain the current two separate feature extraction paths; no trunk-sharing
change, offset bound, auxiliary loss, teacher, or paired augmentation is introduced.

## LDConv fixes

1. Initialize offset-convolution bias and weight to zero.
2. Derive bilinear weights from fractions of unclipped floor coordinates, while
   clamping gather indices. Coincident border neighbors no longer amplify a constant.
3. Replace ineffective backward hook with optimizer parameter groups for offset
   weights/biases, keeping respective decay types and using `samx_offset_lr_scale=0.1`.
   Scheduler base learning rates and bias warmup preserve this ratio. The no-op hook
   symbol remains available for old module pickle loading.
4. Use FP32 coordinates under AMP and cast interpolation weights to feature dtype.

Old LDConv checkpoints remain loadable but boundary predictions can change after
the fix; exact historical LDConv inference is not promised. Old CFSAv1 paths without
LDConv remain intact.

SAMX is attached after normal model/head construction. The added module uses a forked
CPU RNG stream, keeping common initial tensors and the later RNG stream aligned
between ablations. Framework BN/activation initialization is applied to the new block.

## Entry points and execution

`seg_train.py` defaults to experiment 2. `samx_experiment_config.py` holds shared
training options. CFSA and CFSAv1 are explicitly disabled, including P4, ramp and warmup.
The original repaired mode remains selectable but is not in the current queue.

```powershell
# Sequential queue: experiment 2 -> independent FP32 validation -> experiment 3 -> validation.
D:/anaconda3/envs/vajrav1/python.exe -u run_samx_experiments.py --queue
```

Do not start a second queue while one is running. Existing formal experiment directories
are not overwritten or silently renamed. A worker failure or source change stops the
queue and records a failed status. The manifest captures relevant code/config hashes.

Public train settings: 200 epochs, patience=30, seed=42, img_size=1600, rect=True,
batch=32, workers=0, AMP, scratch initialization. Explicit SGD lr=0.01, momentum=0.9,
warmup_bias_lr=0 matches the previous long-run optimizer=auto effective selection.
MABD is disabled (existing BCE path).

Best checkpoints are independently evaluated on all 1532 val images / 3942 instances:
FP32, batch=8, rect=True, img_size=1600, conf=.001, IoU=.7, max_det=300, no TTA.
Training CSV and independent FP32 metrics are kept separate. P/R are validator-selected
operating points; neither these nor one-seed AP differences establish statistical significance.

Audit: `output/samx_experiments_20261027/`. Important files:

- `执行清单.md`: scope and checklist, including the skipped original control.
- `before_samx.zip`: pre-edit code backup.
- `tests_samx.log`, `tests_cfsav1_regression.log`: 9 SAMX + 39 regression checks.
- `smoke_*_startup.json`, `smoke_*_result.json`: separate GPU preflight runs.
- `status.json`: queue and current-worker status; individual `*_progress.json` files.
- `stage_global.log`, `local_global.log`: formal training logs.
- `*_result.json`: best-checkpoint independent validation, counts and SHA256.
- `comparison.json`, `comparison.csv`, `结果对比.md`: generated after both formal runs.

There is no new repaired-original control training, per the user's latest instruction.
Comparisons to uncorrected historical SAMX cannot isolate gate changes from the fixes.
