"""Opt-in P3 mask experiments; original heads and free sampling stay available."""
import math

import torch

from vajra.nn.head import Detection, Segmentation
from vajra.nn.samx_followups import ProtoGeometryEnhance
from vajra.nn.samx_geometry import GeometricDifference


class BoundedGeometricDifference(GeometricDifference):
    """Bound each offset coordinate in P3 cells, with unit derivative at zero."""
    def __init__(self, channels, num_param=3, bound=2.0):
        if isinstance(bound, bool) or not math.isfinite(bound) or bound <= 0:
            raise ValueError('Offset bound must be a positive finite number')
        super().__init__(channels, num_param)
        self.offset_bound = float(bound)

    def _get_p(self, offset, dtype):
        offset = offset.float()
        bound = self.offset_bound
        return super()._get_p(bound * torch.tanh(offset / bound), offset.type())


class CoupledProtoSegmentation(Segmentation):
    """Reuse enhanced P3 for Proto and only the P3 mask-coefficient branch.

    Detection and P4/P5 coefficients still read their original features. The
    enhancement is evaluated once; all branch parameters remain independent.
    """
    def forward(self, x):
        x = list(x)
        proto_input = self.proto_enhance(x[0])
        proto_masks = self.proto(proto_input)
        batch_size = proto_masks.shape[0]
        coefficient_inputs = [proto_input, *x[1:]]
        coefficients = torch.cat([
            branch(feature).view(batch_size, self.num_masks, -1)
            for branch, feature in zip(self.branch_seg, coefficient_inputs)
        ], 2)
        detections = Detection.forward(self, x)
        if self.training:
            return detections, coefficients, proto_masks
        if self.export:
            return torch.cat([detections, coefficients], 1), proto_masks
        return torch.cat([detections[0], coefficients], 1), (detections[1], coefficients, proto_masks)


def configure_proto_mask_variants(head, share_coeff=False, offset_bound=0.0):
    """Install before optimizer construction, preserving weights and RNG state."""
    if not isinstance(share_coeff, bool):
        raise TypeError('share_coeff must be bool')
    if isinstance(offset_bound, bool) or not math.isfinite(offset_bound) or offset_bound < 0:
        raise ValueError('offset_bound must be finite and nonnegative')
    if not share_coeff and offset_bound == 0:
        return
    enhancement = head.proto_enhance
    if not isinstance(enhancement, ProtoGeometryEnhance) or enhancement.mode != 'free':
        raise ValueError('Mask variants require samx_proto_mode="free"')
    if type(head) is not Segmentation:
        raise ValueError('Mask variants require the standard Segmentation head')
    if offset_bound:
        old = enhancement.geometry
        with torch.random.fork_rng(devices=[]):
            bounded = BoundedGeometricDifference(old.inc, old.num_param, offset_bound)
        parameter = next(old.parameters())
        bounded.to(device=parameter.device, dtype=parameter.dtype)
        bounded.load_state_dict(old.state_dict())
        enhancement.geometry = bounded
    if share_coeff:
        # Preserve the existing module graph and tensor identities. Full-object
        # checkpoints record this importable subclass; state-dict keys stay put.
        head.__class__ = CoupledProtoSegmentation
