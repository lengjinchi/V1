"""MABDv1: inner-boundary-conditioned weights and an expanded boundary ROI.

The main BCE/Dice retain their original GT-box support. Positive-anchor
aggregation is handled by SegmentationLoss and is unchanged.
"""
import torch
from torch.nn import functional as F

from vajra.ops import crop_mask, xyxy_to_xywh


def erode_zero_padded(masks, radius=1):
    """Binary erosion with explicit zero background outside the mask canvas."""
    padded = F.pad(masks[:, None], (radius,) * 4, value=0.0)
    return -F.max_pool2d(-padded, 2 * radius + 1, stride=1)[:, 0]


def expanded_boxes(boxes, radius, height, width):
    result = boxes.detach().float().clone()
    result[:, :2] -= radius
    result[:, 2:] += radius
    result[:, 0::2].clamp_(0, width)
    result[:, 1::2].clamp_(0, height)
    return result


def mabd_v1_loss(gt_masks, coefficients, prototypes, boxes, radius=1, eps=1e-6,
                 expand_boundary_roi=True, dice_shape_factor=None,
                 boundary_base=0.05, boundary_gain=0.15, boundary_cap=0.20):
    """Return per-positive losses plus detached last-image diagnostic scalars.

    Reconstruction follows AMP; all probabilities, morphology, BCE and spatial
    reductions are FP32. Thinness always uses a one-pixel inner band. ``radius``
    controls the boundary supervision band and, when enabled, its expanded box.
    Disabling ``expand_boundary_roi`` preserves the band and both weights;
    only boundary supervision is cropped back to the original GT box.
    Optional ``dice_shape_factor`` restores the original MABD Dice weighting
    from its existing morphology_factors implementation. Boundary weighting,
    morphology and FP32 reductions are unaffected.
    """
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError('MABDv1 boundary radius must be a positive integer')
    logits = torch.einsum('in,nhw->ihw', coefficients, prototypes).float()
    gt = gt_masks.detach().float()
    boxes = boxes.detach().float()
    area = xyxy_to_xywh(boxes)[..., 2:].prod(-1).clamp_min(eps)
    pixel_bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
    bce = crop_mask(pixel_bce, boxes).sum((1, 2)) / area

    prediction = crop_mask(logits.sigmoid(), boxes)
    target = crop_mask(gt, boxes)
    intersection = (prediction * target).sum((1, 2))
    dice = 1.0 - (2.0 * intersection + eps) / (prediction.sum((1, 2)) + target.sum((1, 2)) + eps)

    eroded_one = erode_zero_padded(gt)
    inner = (gt - eroded_one).clamp(0, 1)
    gt_area = gt.sum((1, 2))
    thinness = (inner.sum((1, 2)) / gt_area.clamp_min(eps)).clamp(0, 1)
    dice_factor = thinness if dice_shape_factor is None else dice_shape_factor.detach().float()
    if dice_factor.shape != thinness.shape:
        raise ValueError('Dice shape factor must have one value per positive mask')
    dice_weight = 0.30 + 0.20 * dice_factor
    boundary_weight = (boundary_base + boundary_gain * thinness).clamp(0, boundary_cap)

    eroded = eroded_one if radius == 1 else erode_zero_padded(gt, radius)
    dilated = F.max_pool2d(gt[:, None], 2 * radius + 1, stride=1, padding=radius)[:, 0]
    band = (dilated - eroded).clamp(0, 1)
    boundary_boxes = expanded_boxes(boxes, radius, *gt.shape[-2:]) if expand_boundary_roi else boxes
    valid_band = crop_mask(band, boundary_boxes)
    boundary = (pixel_bce * valid_band).sum((1, 2)) / valid_band.sum((1, 2)).clamp_min(eps)
    losses = bce + dice_weight * dice + boundary_weight * boundary

    # Scalar diagnostics do not retain the backward graph or full mask tensors.
    outer = (valid_band - crop_mask(valid_band, boxes)).clamp_min(0)
    diagnostics = dict(
        positives=gt_area.new_tensor(gt.shape[0]),
        empty_gt_masks=(gt_area == 0).sum(), thinness_mean=thinness.mean(),
        dice_weight_mean=dice_weight.mean(), boundary_weight_mean=boundary_weight.mean(),
        bce_mean=bce.mean(), weighted_dice_mean=(dice_weight * dice).mean(),
        weighted_boundary_mean=(boundary_weight * boundary).mean(),
        outer_boundary_pixels=outer.sum(),
    )
    return losses, {key: value.detach() for key, value in diagnostics.items()}
