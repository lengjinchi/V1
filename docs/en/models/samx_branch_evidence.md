# SAMX branch-output evidence experiment

`samx_mode="branch_evidence"` selects the new independent class
`vajra.nn.samx_branch_evidence.BranchEvidenceSAMX`. Original modes remain
available; the default remains `off`. Only nano neck3/P4 is replaced.

## Mechanism

Retain both original feature extraction paths, the original global/local logits,
and the split/cascade/concatenation structure. For each of the two stages:

```text
R_j = regular_path_j(X_j)
D_j = deform_path_j(X_j)
e_j = concat(mean_c(R_j), max_c(R_j), mean_c(D_j), max_c(D_j),
             mean_c(abs(R_j-D_j)), max_c(abs(R_j-D_j)))
alpha_j = softmax(old_global_local_logits + evidence_route(e_j))
Y_j = alpha_j[0] * R_j + alpha_j[1] * D_j
```

The added route is Conv(6,8,3×3) → SiLU → Conv(8,2,1×1), 458 parameters total.
The same route instance is evaluated on each stage's own branch outputs. Both
paths are computed exactly once per stage. Channel statistics are reduced in
FP32, then cast to feature dtype for AMP/explicit-half compatibility.

The last route layer is zero initialized: common initial tensors and outputs
match LocalGlobal exactly in FP32. The extra constructor uses a forked RNG
stream. The new output layer receives a nonzero task gradient at initialization;
earlier route layers begin receiving gradients after it changes. Original
sampling, offset-LR multiplier, feature paths, and task loss remain unchanged.
There is no forced interpretation of large disagreement as noise.

## Experiments

Fixed settings: scratch, 200 epochs, patience=30, img_size=1600, rect=True,
batch=32, workers=0, AMP, seed=42, deterministic=True, SGD lr0=.01,
momentum=.9, warmup_bias_lr=0, offset LR multiplier=.1. CFSA, CFSAv1, MABD,
pair/reference supervision, freezing and gate-ramp schedules remain disabled.

- G1: new P4 route, Proto enhancement off. Compare with saved LocalGlobal.
- G2: new P4 route plus saved-design P3 free geometry, c16/gain_init=.1.
  Compare with G1 and saved P3 free design. Independently trained from scratch.

G2 scheduling is fixed before G1: Mask AP50:95 gain at least .30 percentage
points against LocalGlobal, and no more than .10 pp loss for Mask AP50, Mask
Recall, or boundary F1. This is a practical progression rule, not a statistical
significance claim. G2 is skipped with a saved explanation otherwise.

## Run and artifacts

```powershell
D:/anaconda3/envs/vajrav1/python.exe -u run_samx_branch_experiments.py --preflight
D:/anaconda3/envs/vajrav1/python.exe -u run_samx_branch_experiments.py --queue
```

Preflight requires the passed test manifest, reproduces both old controls and
runs both 5%-dataset, one-epoch smoke trials with full validation. The larger
smoke allows ten batches for default AMP scaling to settle while checking live
gate gradients; formal training settings are unchanged. Duplicate
directories are rejected. No old model or checkpoint is overwritten.

Audit directory: `output/samx_branch_evidence_20261031/`. `status.json` records
queue state; individual `*_progress.json` records epoch/loss/metrics and worker
PID. Source snapshots and hashes are checked between children and after runs;
102 existing checkpoint files were protected before integration.

Main results use the unchanged combined Box+Mask-fitness best checkpoint,
independently validated on 1532 images / 3942 instances, FP32/batch8, with the
same boundary protocol as previous trials. An additional Mask-AP95-best file is
saved for later analysis but does not replace the main result or stopping rule.

Each trial also runs full same-checkpoint validation with only the newly added
logit correction zeroed using temporary hooks. Learned gate statistics exclude
padding and report max-projected GT foreground vs background. Gate variance or
heatmaps alone do not establish task utility. Fixed-conf=.25 error counts are
micro counts and must not be confused with official macro P/R/AP.

Speed is measured using same-process alternating model order, fused FP32,
batch1, 288×1632, 50 initial warmups, then three rounds of 10 warmup and 100
timed forwards per model. It excludes loading, NMS and mask reconstruction.

The cancelled repaired-original SAMX training is not scheduled. One-seed gains
are provisional; branch-output gating alone is not asserted to be novel.
