"""CFSAv1: local band modulation with optional global and P4 context.

The original FFTGuidedEMA/CFSA remains in modules.py. This module uses no FFT,
downsampling, batch statistics, reference images, or teacher at inference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalBandDecomposition(nn.Module):
    """Three overlapping bands on the input grid; their sum is the input."""

    def __init__(self):
        super().__init__()
        for size, sigma in ((3, 0.8), (5, 1.2)):
            axis = torch.arange(size, dtype=torch.float32) - size // 2
            kernel = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2 * sigma**2))
            self.register_buffer(f"kernel{size}", (kernel / kernel.sum())[None, None])

    def _smooth(self, x, size):
        kernel = getattr(self, f"kernel{size}").to(dtype=x.dtype)
        kernel = kernel.expand(x.shape[1], 1, size, size)
        # Replicate padding also supports very small/odd feature maps.
        pad = size // 2
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, groups=x.shape[1])

    def forward(self, x):
        fine, coarse = self._smooth(x, 3), self._smooth(x, 5)
        return torch.stack((coarse, fine - coarse, x - fine), dim=2)  # B,C,3,H,W


def band_energy(bands, groups):
    """Grouped, locally averaged absolute responses, ordered group then band."""
    b, c, _, h, w = bands.shape
    energy = bands.float().reshape(b, groups, c // groups, 3, h, w).abs().mean(2)
    return F.avg_pool2d(energy.reshape(b, groups * 3, h, w), 3, 1, 1, count_include_pad=False)


class CFSAv1(nn.Module):
    """Identity-initialized, spatially varying group/band gains.

    New models use only task supervision. Old sensitivity-head checkpoints keep
    a minimal inference-only compatibility path; no teacher or paired training.
    """

    def __init__(self, channels, groups=4, context_channels=16, rho=0.25,
                 gate_smooth=False, global_context=False, semantic_channels=0):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("CFSAv1 channels must be a positive integer")
        if not isinstance(groups, int) or groups <= 0 or channels % groups:
            raise ValueError(f"CFSAv1 channels={channels} must be divisible by groups={groups}")
        if not isinstance(context_channels, int) or context_channels <= 0:
            raise ValueError("CFSAv1 context_channels must be a positive integer")
        if not math.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("CFSAv1 rho must be strictly between 0 and 1")
        if isinstance(semantic_channels, bool) or not isinstance(semantic_channels, int) or semantic_channels < 0:
            raise ValueError("CFSAv1 semantic_channels must be a non-negative integer")
        self.channels, self.groups = channels, groups
        self.context_channels, self.rho = context_channels, rho
        self.gate_smooth = gate_smooth
        self.global_context = global_context
        self.semantic_channels = semantic_channels
        # Scheduled by the trainer, persisted for identical validation/inference.
        # One preserves the behaviour of checkpoints trained without a ramp.
        self.register_buffer("gate_scale", torch.tensor(1.0))
        self.bands = LocalBandDecomposition()
        self.context = nn.Sequential(
            nn.Conv2d(channels, context_channels, 1), nn.SiLU(),
            nn.Conv2d(context_channels, context_channels, 3, padding=1, groups=context_channels), nn.SiLU(),
        )
        condition_channels = context_channels + 3 * groups
        self.gate = nn.Sequential(
            nn.Conv2d(condition_channels, context_channels, 1),
            nn.SiLU(), nn.Conv2d(context_channels, 3 * groups, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # Pool before the MLP: one group/band logit per image, broadcast over
        # spatial positions. No batch statistics or additional spatial filters.
        self.global_gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(condition_channels, context_channels, 1),
                nn.SiLU(), nn.Conv2d(context_channels, 3 * groups, 1),
            ) if global_context else None
        )
        if self.global_gate is not None:
            nn.init.zeros_(self.global_gate[-1].weight)
            nn.init.zeros_(self.global_gate[-1].bias)
        # Compute hints on the coarse P4 grid, then resize only the 3*groups
        # logits. P3 bands and the detection/coefficient branches are unchanged.
        self.semantic_gate = (
            nn.Sequential(
                nn.Conv2d(semantic_channels, context_channels, 1), nn.SiLU(),
                nn.Conv2d(context_channels, context_channels, 3, padding=1,
                          groups=context_channels), nn.SiLU(),
                nn.Conv2d(context_channels, 3 * groups, 1),
            ) if semantic_channels else None
        )
        if self.semantic_gate is not None:
            nn.init.zeros_(self.semantic_gate[-1].weight)
            nn.init.zeros_(self.semantic_gate[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Permit strict resume/EMA loading of pre-ramp CFSAv1 state dictionaries.
        if prefix + "gate_scale" not in state_dict:
            state_dict = state_dict.copy()
            state_dict[prefix + "gate_scale"] = torch.ones_like(self.gate_scale)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, semantic=None):
        semantic_gate = getattr(self, "semantic_gate", None)
        if semantic_gate is not None:
            if semantic is None:
                raise ValueError("CFSAv1 P4 semantic context is enabled but no P4 feature was supplied")
            if semantic.ndim != 4 or semantic.shape[0] != x.shape[0] or semantic.shape[1] != self.semantic_channels:
                raise ValueError("CFSAv1 expects P4 with matching batch size and configured semantic channels")
        bands = self.bands(x)
        context = self.context(x)
        energy = band_energy(bands, self.groups).log1p().to(context.dtype)
        condition = torch.cat((context, energy), dim=1)
        # Historical full-module pickles may contain a wider gate and a trained
        # sensitivity head. Preserve their inference rather than silently change it.
        if getattr(self, "use_sensitivity", False):
            if self.training:
                raise RuntimeError("Reference-supervised CFSAv1 training has been removed; legacy inference is supported.")
            sensitivity = self.sensitivity_head(condition.detach()).sigmoid()
            condition = torch.cat((condition, sensitivity.detach()), dim=1)
        logits = self.gate(condition)
        # Missing branch keeps pre-global-context full-module checkpoints valid.
        global_gate = getattr(self, "global_gate", None)
        if global_gate is not None:
            logits = logits + global_gate(condition)
        if semantic_gate is not None:
            semantic_logits = semantic_gate(semantic)
            logits = logits + F.interpolate(semantic_logits, size=x.shape[-2:],
                                            mode="bilinear", align_corners=False)
        # Bound the combined logits once, preserving the original gain range.
        delta = self.rho * logits.tanh()
        # getattr also supports old full-module pickles without this buffer.
        delta = delta * getattr(self, "gate_scale", 1.0)
        # Smooth coefficients, not bands. Missing flag keeps old pickles unchanged.
        if getattr(self, "gate_smooth", False):
            delta = F.avg_pool2d(delta, 3, stride=1, padding=1, count_include_pad=False)
        b, c, h, w = x.shape
        delta = delta.reshape(b, self.groups, 1, 3, h, w)
        grouped = bands.reshape(b, self.groups, c // self.groups, 3, h, w)
        # Algebraically sum((1 + delta) * band), but exactly identity at init.
        out = x + (delta * grouped).sum(3).reshape(b, c, h, w).to(x.dtype)
        return out

    def get_module_info(self):
        return "CFSAv1", str([self.channels, self.groups, self.context_channels, self.rho,
                              getattr(self, "gate_smooth", False),
                              getattr(self, "global_context", False),
                              getattr(self, "semantic_channels", 0)])
