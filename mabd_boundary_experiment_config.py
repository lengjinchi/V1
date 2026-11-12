"""Restore original Dice weights; keep thinness boundary weights and GT ROI."""
from mabd_weights_experiment_config import (
    ROOT, FULL_AUDIT, AUDIT as WEIGHTS_AUDIT,
    training_options as weights_training_options,
    validation_options as weights_validation_options,
)

AUDIT = ROOT / 'output' / 'mabd_v1_boundary_20261110'
NAME = 'MABDv1_origDice_thinBoundary_gtROI_base_img1600_seed42_20261110'


def training_options(smoke=False):
    options = weights_training_options(smoke)
    options.update(
        mask_loss_mode='mabd_v1_boundary',
        project=str(AUDIT / 'smoke') if smoke else str(ROOT / 'MABD compare'),
        name='mabd_v1_boundary_smoke' if smoke else NAME,
    )
    return options


def validation_options(name):
    options = weights_validation_options(name)
    options['project'] = str(AUDIT / 'validation')
    return options
