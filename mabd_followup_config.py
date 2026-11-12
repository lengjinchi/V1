"""Bounded, conditional follow-up plan authorized on 2026-11-10."""
from pathlib import Path

from mabd_boundary_experiment_config import (
    ROOT, AUDIT as PREVIOUS_AUDIT, training_options as previous_training_options,
    validation_options as previous_validation_options,
)

AUDIT = ROOT / 'output' / 'mabd_followup_plan_20261110'
REPEAT_SEEDS = (0, 123)
# These are screening rules, not significance tests; freeze before E1 starts.
SCREEN = dict(min_gain=.003, ap_noninferiority=.001, recall_drop=.01,
              precision_drop=.02, boundary_f1_drop=.005, common_class_ap95_drop=.01)
SCALE_MISMATCH = .10


def training_options(tag, *, seed=42, smoke=False, gain=.15, joint=False, original=False):
    options = previous_training_options(smoke)
    options.update(mask_loss_mode='mabd' if original else 'mabd_v1_boundary',
        mabd_v1_boundary_base=0.0, mabd_v1_boundary_gain=float(gain),
        mabd_v1_boundary_cap=.15, seed=seed,
        project=str(AUDIT / 'smoke') if smoke else str(ROOT / ('Joint compare' if joint else 'MABD compare')),
        name=f'{tag}_smoke' if smoke else f'MABDplan_{tag}_img1600_seed{seed}_20261110')
    if joint:
        options.update(use_cfsav1=True, cfsav1_groups=4, cfsav1_context_channels=16,
            cfsav1_rho=.25, samx_mode='local_global', samx_offset_lr_scale=.1)
    return options


def validation_options(tag):
    options = previous_validation_options(tag)
    options['project'] = str(AUDIT / 'validation')
    return options


def candidate_passes(candidate, reference):
    """Prespecified validation screening; all deltas are proportions."""
    cm, rm = candidate['metrics'], reference['metrics']
    delta = {key: cm[key] - rm[key] for key in (
        'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)',
        'metrics/recall(Mask)', 'metrics/precision(Mask)')}
    ap50, ap95 = delta['metrics/mAP50(Mask)'], delta['metrics/mAP50-95(Mask)']
    boundary = candidate['boundary']['overall']['f1'] - reference['boundary']['overall']['f1']
    common = {key: candidate['per_class'][key]['map95'] - value['map95']
              for key, value in reference['per_class'].items() if value['instances'] >= 100}
    checks = dict(ap50_noninferior=ap50 >= -SCREEN['ap_noninferiority'],
        ap95_noninferior=ap95 >= -SCREEN['ap_noninferiority'],
        at_least_one_ap_gain=max(ap50, ap95) >= SCREEN['min_gain'],
        recall_stable=delta['metrics/recall(Mask)'] >= -SCREEN['recall_drop'],
        precision_stable=delta['metrics/precision(Mask)'] >= -SCREEN['precision_drop'],
        boundary_stable=boundary >= -SCREEN['boundary_f1_drop'],
        common_classes_stable=all(v >= -SCREEN['common_class_ap95_drop'] for v in common.values()))
    return dict(passed=all(checks.values()), checks=checks, metric_deltas=delta,
                boundary_f1_delta=boundary, common_class_ap95_deltas=common)


def bounded_gain(q, target, cap=.15):
    """Train-only mean calibration: solve mean(min(gain*q, cap)) = target."""
    import numpy as np
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Calibration needs finite, nonempty thinness values in [0, 1]')
    if not 0 <= target <= cap or target > cap * np.mean(q > 0) + 1e-12:
        raise ValueError('Target mean is unreachable with this bounded weighting')
    if target == 0:
        return 0.0
    low, high = 0.0, 1.0
    while np.minimum(high*q, cap).mean() < target:
        high *= 2
        if high > 1e8:
            raise ValueError('Ill-conditioned calibration')
    for _ in range(80):
        middle = (low+high)/2
        if np.minimum(middle*q, cap).mean() < target:
            low = middle
        else:
            high = middle
    return (low+high)/2
