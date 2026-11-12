"""Shared configuration for manual and sequential SAMX experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "output" / "samx_experiments_20261027"
GEOMETRY_AUDIT = ROOT / "output" / "samx_geometry_20261028"
EXPERIMENTS = {
    "original": "SAMX_fix1_original_cfsa0_cfsav10_img1600_seed42_20261027",
    "stage_global": "SAMX_fix1_E2_stageGlobal_cfsa0_cfsav10_img1600_seed42_20261027",
    "local_global": "SAMX_fix1_E3_localGlobal_cfsa0_cfsav10_img1600_seed42_20261027",
    "geometry_residual": "SAMX_geometryResidual_P4_gamma0.1_cfsa0_cfsav10_img1600_seed42_20261028",
}


def clear_validation_cache(validator):
    """Release unused training memory before validation on Windows CUDA."""
    if validator.device.type == "cuda":
        import torch
        torch.cuda.empty_cache()


def training_options(mode, smoke=False):
    if mode not in EXPERIMENTS:
        raise ValueError(f"Unknown SAMX experiment: {mode}")
    audit = GEOMETRY_AUDIT if mode == "geometry_residual" else AUDIT
    return dict(
        data=r"F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml",
        epochs=1 if smoke else 200, patience=30, img_size=1600, rect=True,
        workers=0, device=0, batch=32, amp=True, seed=42, deterministic=True,
        pretrained=False, resume=False, fraction=0.01 if smoke else 1.0,
        optimizer="SGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0,
        project=str(audit / "smoke") if smoke else str(ROOT / "SAMX compare"),
        name=("smoke_" + mode) if smoke else EXPERIMENTS[mode],
        exist_ok=False, plots=not smoke,
        samx_mode=mode, samx_offset_lr_scale=0.1, samx_geometry_gain=0.1,
        use_cfsa=False, use_cfsav1=False,
        cfsav1_global_context=False, cfsav1_semantic_context=False,
        cfsav1_gate_smooth=False, cfsav1_gate_ramp_epochs=0, cfsav1_warmup_epochs=0,
    )
