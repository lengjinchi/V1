"""Joint experiment matching the 2026-11-08 presentation; old runs are untouched."""
from pathlib import Path

from samx_experiment_config import clear_validation_cache

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'output' / 'joint_cfsav1_samx_20261109'
PRESENTATION = Path(r'C:\Users\Administrator\Desktop\11.8\zjx_方法汇报精简版_20261108.pptx')
NAME = 'Joint_CFSAv1_local_rho0.25_SAMX_localGlobal_P4_img1600_seed42_20261109'


def training_options(smoke=False):
    return dict(
        data=r'F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml',
        epochs=1 if smoke else 200, patience=30, img_size=1600, rect=True,
        batch=32, workers=0, device=0, amp=True, seed=42, deterministic=True,
        pretrained=False, resume=False, optimizer='SGD', lr0=0.01,
        momentum=0.9, warmup_bias_lr=0.0,
        # Several real optimizer steps check both zero-initialized gates.
        fraction=0.10 if smoke else 1.0,
        project=str(AUDIT / 'smoke') if smoke else str(ROOT / 'Joint compare'),
        name='joint_smoke' if smoke else NAME, exist_ok=False, plots=not smoke,
        use_cfsa=False, use_cfsav1=True,
        cfsav1_groups=4, cfsav1_context_channels=16, cfsav1_rho=0.25,
        cfsav1_global_context=False, cfsav1_semantic_context=False,
        cfsav1_gate_smooth=False, cfsav1_gate_ramp_epochs=0, cfsav1_warmup_epochs=0,
        samx_mode='local_global', samx_offset_lr_scale=0.1,
        samx_proto_mode='off', samx_proto_share_coeff=False,
        samx_proto_offset_bound=0.0,
    )


def validation_options(name):
    return dict(
        data=training_options()['data'], split='val', img_size=1600, rect=True,
        batch=8, workers=0, device=0, half=False, conf=0.001, iou=0.7,
        max_det=300, overlap_mask=True, mask_ratio=4, augment=False,
        plots=False, verbose=False, project=str(AUDIT / 'validation'),
        name=name, exist_ok=False,
    )
