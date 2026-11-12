"""Thinness weights only: the full MABDv1 run with the original boundary ROI."""
from mabd_experiment_config import (
    ROOT, AUDIT as FULL_AUDIT, training_options as full_training_options,
    validation_options as full_validation_options,
)

AUDIT = ROOT / 'output' / 'mabd_v1_weights_20261110'
NAME = 'MABDv1_weightsOnly_gtROI_base_img1600_seed42_20261110'


def training_options(smoke=False):
    options = full_training_options(smoke)
    options.update(
        mask_loss_mode='mabd_v1_weights',
        project=str(AUDIT / 'smoke') if smoke else str(ROOT / 'MABD compare'),
        name='mabd_v1_weights_smoke' if smoke else NAME,
    )
    return options


def validation_options(name):
    options = full_validation_options(name)
    options['project'] = str(AUDIT / 'validation')
    return options
