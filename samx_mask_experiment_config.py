"""Two independent scratch trials against the completed proto_free model."""
import math

from samx_followup_config import options as previous_options
from samx_experiment_config import ROOT

AUDIT = ROOT / 'output/samx_mask_experiments_20261030'
PREVIOUS_AUDIT = ROOT / 'output/samx_followups_20261029'
BASELINE_RESULT = PREVIOUS_AUDIT / 'proto_free_result.json'
MODES = ('proto_coeff_shared', 'proto_offset_bounded')


def select_offset_bound(p95_absolute):
    """Choose the smallest power-of-two P3 limit covering the observed p95.

    This is a predeclared heuristic from training-image statistics, not an AP
    search or a claim that offsets below the limit are numerically unchanged.
    """
    if not math.isfinite(p95_absolute) or p95_absolute < 0:
        raise ValueError('p95 must be finite and nonnegative')
    return float(2 ** max(0, math.ceil(math.log2(max(p95_absolute, 1.)))))


def options(mode, smoke=False, bound=None):
    if mode not in MODES:
        raise ValueError(mode)
    if mode == 'proto_offset_bounded' and (bound is None or bound <= 0):
        raise ValueError('Freeze the diagnostic-selected bound before training')
    args = previous_options('proto_free', smoke)
    suffix = 'protoCoeffShared' if mode == MODES[0] else f'protoBoundR{bound:g}'
    args.update(samx_proto_share_coeff=mode == MODES[0],
                samx_proto_offset_bound=float(bound) if mode == MODES[1] else 0.0,
                project=str(AUDIT / 'smoke') if smoke else str(ROOT / 'SAMX compare'),
                name=('smoke_' + mode) if smoke else
                     f'SAMX_localGlobalP4_{suffix}_P3_c16_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261030',
                exist_ok=False)
    return args


def validation_options(name):
    return dict(data=previous_options('proto_free')['data'], split='val', img_size=1600,
                rect=True, batch=8, workers=0, device=0, half=False, conf=.001,
                iou=.7, max_det=300, overlap_mask=True, mask_ratio=4,
                augment=False, plots=False, verbose=False,
                project=str(AUDIT / 'validation'), name=name, exist_ok=False)
