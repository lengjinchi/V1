"""Standalone MABDv1 full variant; reuse, never retrain, the historical MABD."""
from pathlib import Path

from samx_experiment_config import clear_validation_cache

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'output' / 'mabd_v1_full_20261109'
ORIGINAL_RUN = ROOT / 'runs3/steel segment Vajra-new improvebceloss/Vajra-new loss'
ORIGINAL_CHECKPOINT = ORIGINAL_RUN / 'weights/best-vajra-v1-nano-seg.pt'
NAME = 'MABDv1_full_thinness_roi1_base_img1600_seed42_20261109'


def training_options(smoke=False):
    return dict(
        data=r'F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml',
        # Historical MABD-only run used 300; use its training horizon/schedule.
        epochs=1 if smoke else 300, patience=30, img_size=1600, rect=True,
        batch=32, workers=0, device=0, amp=True, seed=42, deterministic=True,
        pretrained=False, resume=False,
        # Historical auto selects SGD(.01,.9) at 300 full epochs and sets
        # warmup_bias_lr=0. Explicit SGD also keeps the smoke check on SGD.
        optimizer='SGD', lr0=0.01, momentum=0.9, warmup_bias_lr=0.0,
        warmup_epochs=3.0, warmup_momentum=0.8,
        fraction=0.10 if smoke else 1.0,
        project=str(AUDIT / 'smoke') if smoke else str(ROOT / 'MABD compare'),
        name='mabd_v1_full_smoke' if smoke else NAME,
        exist_ok=False, plots=not smoke,
        mask_loss_mode='mabd_v1', mabd_v1_boundary_radius=1,
        box=7.5, seg=7.5, cls=0.5, dfl=1.5,
        overlap_mask=True, mask_ratio=4,
        use_cfsa=False, use_cfsav1=False, samx_mode='off',
        samx_proto_mode='off', samx_proto_share_coeff=False, samx_proto_offset_bound=0.0,
        cfsav1_global_context=False, cfsav1_semantic_context=False,
        cfsav1_gate_smooth=False, cfsav1_gate_ramp_epochs=0, cfsav1_warmup_epochs=0,
    )


def validation_options(name):
    return dict(
        data=training_options()['data'], split='val', img_size=1600, rect=True,
        batch=8, workers=0, device=0, half=False, conf=0.001, iou=0.7,
        max_det=300, overlap_mask=True, mask_ratio=4, augment=False,
        plots=False, verbose=False, project=str(AUDIT / 'validation'),
        name=name, exist_ok=False,
    )
