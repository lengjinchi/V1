"""Predeclared G1, then conditional G2; both are scratch, single-factor trials."""
from samx_experiment_config import ROOT, training_options
from samx_mask_experiment_config import validation_options as previous_validation_options

AUDIT = ROOT / "output/samx_branch_evidence_20261031"
MODES = ("g1_p4_only", "g2_p4_proto_free")
BASELINES = {
    "g1_p4_only": ROOT / "output/samx_followups_20261029/baseline_local_global_result.json",
    "g2_p4_proto_free": ROOT / "output/samx_mask_experiments_20261030/baseline_result.json",
}
EXPERIMENTS = {
    "g1_p4_only": "SAMX_G1_branchEvidenceP4_protoOff_cfsa0_cfsav10_img1600_seed42_20261031",
    "g2_p4_proto_free": "SAMX_G2_branchEvidenceP4_protoFreeP3_c16_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261031",
}

# Practical scheduling rule, fixed before training; not a significance test.
# AP95 must gain >=0.30 percentage points. No other metric may lose >0.10 pp.
G2_MIN_AP95_GAIN = 0.003
G2_MAX_OTHER_DROP = 0.001


def options(mode, smoke=False):
    if mode not in MODES:
        raise ValueError(mode)
    args = training_options("local_global", smoke=smoke)
    args.update(
        samx_mode="branch_evidence",
        samx_proto_mode="off" if mode == MODES[0] else "free",
        samx_proto_channels=16, samx_proto_gain=0.1,
        samx_proto_share_coeff=False, samx_proto_offset_bound=0.0,
        project=str(AUDIT / "smoke") if smoke else str(ROOT / "SAMX compare"),
        name=("smoke_" + mode) if smoke else EXPERIMENTS[mode],
        exist_ok=False,
    )
    if smoke:
        # Ten batches allow default AMP loss scaling to settle on real 1600px
        # data, instead of concluding that a two-batch transient is a dead gate.
        args["fraction"] = 0.05
    return args


def validation_options(name):
    args = previous_validation_options(name)
    args.update(project=str(AUDIT / "validation"), name=name, exist_ok=False)
    return args


def decide_g2(baseline, candidate):
    metrics = ("metrics/mAP50-95(Mask)", "metrics/mAP50(Mask)", "metrics/recall(Mask)")
    deltas = {key: candidate["metrics"][key] - baseline["metrics"][key] for key in metrics}
    deltas["boundary_f1"] = candidate["boundary"]["overall"]["f1"] - baseline["boundary"]["overall"]["f1"]
    checks = {
        "mask_ap95_gain_ge_0.30pp": deltas[metrics[0]] >= G2_MIN_AP95_GAIN - 1e-12,
        "mask_ap50_drop_le_0.10pp": deltas[metrics[1]] >= -G2_MAX_OTHER_DROP - 1e-12,
        "mask_recall_drop_le_0.10pp": deltas[metrics[2]] >= -G2_MAX_OTHER_DROP - 1e-12,
        "boundary_f1_drop_le_0.10pp": deltas["boundary_f1"] >= -G2_MAX_OTHER_DROP - 1e-12,
    }
    return dict(run_g2=all(checks.values()), checks=checks,
                deltas_percentage_points={key: 100 * value for key, value in deltas.items()},
                note="Predeclared scheduling threshold; one seed does not establish statistical significance.")
