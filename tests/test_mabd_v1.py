import io
import math
import unittest

import torch
from torch import nn

from vajra.configs import get_config
from vajra.core.model import Model
from vajra.loss import SegmentationLoss
from vajra.loss.mabd_v1 import expanded_boxes, mabd_v1_loss


class DummyModel(nn.Module):
    def __init__(self, mode='bce'):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        head = nn.Identity()
        head.reg_max = 16
        self.model = nn.ModuleList([head])
        self.num_classes = 4
        self.stride = torch.tensor([8., 16., 32.])
        self.args = get_config(model_configuration={'mask_loss_mode': mode})


class MABDv1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def sample(device='cpu', dtype=torch.float32):
        gt = torch.zeros(1, 10, 12, device=device)
        gt[:, 3:7, 4:8] = 1
        proto = torch.zeros(1, 10, 12, device=device, dtype=dtype, requires_grad=True)
        coeff = torch.ones(1, 1, device=device, dtype=dtype, requires_grad=True)
        boxes = torch.tensor([[4., 3., 8., 7.]], device=device)
        return gt, coeff, proto, boxes

    def test_original_bce_and_mabd_still_match_analytic_rectangle(self):
        sample = self.sample()
        gt, coeff, proto, boxes = sample
        for mode, expected in [('bce', math.log(2)),
                               ('mabd', math.log(2) + .3 * (1 - (16 + 1e-6)/(24 + 1e-6)))]:
            criterion = SegmentationLoss(DummyModel(mode))
            value = criterion.single_mask_loss(gt, coeff, proto, boxes)
            self.assertAlmostEqual(value.item(), expected, places=6)
            self.assertEqual(criterion.use_morph_mask_loss, mode != 'bce')

    def test_inner_band_weights_and_loss_have_known_values(self):
        values, stats = mabd_v1_loss(*self.sample())
        # 4x4 rectangle, 2x2 eroded interior: q=12/16=0.75.
        self.assertAlmostEqual(stats['thinness_mean'].item(), .75, places=6)
        self.assertAlmostEqual(stats['dice_weight_mean'].item(), .45, places=6)
        self.assertAlmostEqual(stats['boundary_weight_mean'].item(), .1625, places=6)
        dice = 1 - (16 + 1e-6)/(24 + 1e-6)
        self.assertAlmostEqual(values.item(), math.log(2)*(1+.1625)+.45*dice, places=6)
        self.assertGreater(stats['outer_boundary_pixels'].item(), 0)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

    def test_expanded_boundary_produces_only_local_outer_gradients(self):
        gt, coeff, proto, boxes = self.sample()
        values, _ = mabd_v1_loss(gt, coeff, proto, boxes)
        values.sum().backward()
        self.assertGreater(proto.grad[0, 4, 3].item(), 0)  # one pixel outside box
        self.assertEqual(proto.grad[0, 4, 2].item(), 0)  # beyond the expanded ROI
        self.assertLess(proto.grad[0, 4, 4].item(), 0)  # foreground needs higher logit
        _, _, old_proto, _ = self.sample()
        SegmentationLoss(DummyModel('mabd')).single_mask_loss(gt, coeff, old_proto, boxes).backward()
        self.assertEqual(old_proto.grad[0, 4, 3].item(), 0)

    def test_expansion_clamps_without_mutating_original_boxes(self):
        boxes = torch.tensor([[0., 0., 3., 2.], [9., 7., 12., 10.]])
        original = boxes.clone()
        actual = expanded_boxes(boxes, 1, 10, 12)
        torch.testing.assert_close(actual, torch.tensor([[0., 0., 4., 3.], [8., 6., 12., 10.]]))
        torch.testing.assert_close(boxes, original)

    def test_weights_only_has_no_outer_gradients_but_retains_inner_supervision(self):
        gt, coeff, proto, boxes = self.sample()
        values, stats = mabd_v1_loss(gt, coeff, proto, boxes, expand_boundary_roi=False)
        values.sum().backward()
        self.assertEqual(stats['outer_boundary_pixels'].item(), 0)
        self.assertEqual(proto.grad[0, 4, 3].item(), 0)
        self.assertEqual(proto.grad[0, 4, 8].item(), 0)
        self.assertLess(proto.grad[0, 4, 4].item(), 0)
        self.assertGreater(coeff.grad.abs().sum().item() + proto.grad.abs().sum().item(), 0)

    def test_weights_only_changes_only_boundary_support(self):
        gt, coeff, proto, boxes = self.sample()
        with torch.no_grad():
            proto[:, :, :4] = 3.0  # nonuniform background logits expose the ROI difference
        full, full_stats = mabd_v1_loss(gt, coeff, proto, boxes)
        weights, weights_stats = mabd_v1_loss(gt, coeff, proto, boxes, expand_boundary_roi=False)
        for key in ('thinness_mean', 'dice_weight_mean', 'boundary_weight_mean',
                    'bce_mean', 'weighted_dice_mean'):
            torch.testing.assert_close(full_stats[key], weights_stats[key], rtol=0, atol=0)
        torch.testing.assert_close(full.mean() - weights.mean(),
            full_stats['weighted_boundary_mean'] - weights_stats['weighted_boundary_mean'])
        self.assertGreater(full_stats['weighted_boundary_mean'], weights_stats['weighted_boundary_mean'])

    def test_full_variant_default_matches_explicit_expansion(self):
        implicit, implicit_stats = mabd_v1_loss(*self.sample())
        explicit, explicit_stats = mabd_v1_loss(*self.sample(), expand_boundary_roi=True)
        torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)
        for key in implicit_stats:
            torch.testing.assert_close(implicit_stats[key], explicit_stats[key], rtol=0, atol=0)

    def test_weights_experiment_configuration_is_single_ablation(self):
        from mabd_experiment_config import training_options as full_options
        from mabd_weights_experiment_config import training_options as weights_options
        full, weights = full_options(), weights_options()
        diff = {key for key in full.keys() | weights.keys() if full.get(key) != weights.get(key)}
        self.assertEqual(diff, {'mask_loss_mode', 'name'})
        self.assertEqual(weights['mabd_v1_boundary_radius'], 1)
        self.assertFalse(weights['use_cfsa'] or weights['use_cfsav1'])
        self.assertEqual(weights['samx_mode'], 'off')

    def test_boundary_only_restores_original_dice_on_square_and_long_boxes(self):
        criterion = SegmentationLoss(DummyModel('mabd_v1_boundary'))
        for box in ([4., 3., 8., 7.], [2., 3., 10., 7.]):
            gt, coeff, proto, _ = self.sample()
            boxes = torch.tensor([box])
            gt.zero_()
            gt[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
            loss = criterion.single_mask_loss(gt, coeff, proto, boxes)
            stats = criterion.mabd_v1_last_diagnostics
            ratio = max((box[2]-box[0])/(box[3]-box[1]), (box[3]-box[1])/(box[2]-box[0]))
            expected_weight = .3 + .2 * min(math.log(ratio)/math.log(50), 1)
            self.assertAlmostEqual(stats['dice_weight_mean'].item(), expected_weight, places=6)
            self.assertAlmostEqual(stats['original_dice_weight_mean'].item(), expected_weight, places=6)
            self.assertEqual(stats['outer_boundary_pixels'].item(), 0)
            self.assertTrue(torch.isfinite(loss))

    def test_boundary_only_differs_from_previous_variant_only_in_dice_weight(self):
        sample = self.sample()
        previous = SegmentationLoss(DummyModel('mabd_v1_weights'))
        boundary = SegmentationLoss(DummyModel('mabd_v1_boundary'))
        previous_loss = previous.single_mask_loss(*sample)
        boundary_loss = boundary.single_mask_loss(*sample)
        old_stats, new_stats = previous.mabd_v1_last_diagnostics, boundary.mabd_v1_last_diagnostics
        for key in ('thinness_mean', 'boundary_weight_mean', 'bce_mean',
                    'weighted_boundary_mean', 'outer_boundary_pixels'):
            torch.testing.assert_close(old_stats[key], new_stats[key], rtol=0, atol=0)
        torch.testing.assert_close(previous_loss - boundary_loss,
            old_stats['weighted_dice_mean'] - new_stats['weighted_dice_mean'])
        self.assertGreater(previous_loss.item(), boundary_loss.item())

    def test_boundary_only_preserves_original_gt_box_gradients(self):
        gt, coeff, proto, boxes = self.sample()
        criterion = SegmentationLoss(DummyModel('mabd_v1_boundary'))
        criterion.single_mask_loss(gt, coeff, proto, boxes).backward()
        self.assertEqual(proto.grad[0, 4, 3].item(), 0)
        self.assertEqual(proto.grad[0, 4, 8].item(), 0)
        self.assertLess(proto.grad[0, 4, 4].item(), 0)
        self.assertTrue(torch.isfinite(proto.grad).all())

    def test_boundary_only_matches_original_except_weight_on_interior_mask(self):
        from torch.nn import functional as F
        from vajra.ops import crop_mask, xyxy_to_xywh
        gt, coeff, proto, boxes = self.sample()
        gt[:, 5, 5] = 0
        with torch.no_grad():
            proto.copy_(torch.linspace(-2, 2, proto.numel()).reshape_as(proto))
        original = SegmentationLoss(DummyModel('mabd'))
        boundary = SegmentationLoss(DummyModel('mabd_v1_boundary'))
        original_loss = original.single_mask_loss(gt, coeff, proto, boxes)
        boundary_loss = boundary.single_mask_loss(gt, coeff, proto, boxes)
        stats = boundary.mabd_v1_last_diagnostics
        _, original_factor = original.morphology_factors(gt, boxes, xyxy_to_xywh(boxes)[..., 2:].prod(-1))
        band = crop_mask(original.boundary_map(gt), boxes)
        pixel_bce = F.binary_cross_entropy_with_logits(torch.einsum('in,nhw->ihw', coeff, proto), gt, reduction='none')
        boundary_bce = (pixel_bce * band).sum((1, 2)) / band.sum((1, 2))
        expected_delta = (stats['boundary_weight_mean'] - .15 * original_factor) * boundary_bce
        torch.testing.assert_close(boundary_loss - original_loss, expected_delta.sum(), atol=1e-6, rtol=1e-5)

    def test_boundary_experiment_is_single_ablation_of_previous_run(self):
        from mabd_weights_experiment_config import training_options as previous_options
        from mabd_boundary_experiment_config import training_options as boundary_options
        previous, boundary = previous_options(), boundary_options()
        diff = {key for key in previous.keys() | boundary.keys() if previous.get(key) != boundary.get(key)}
        self.assertEqual(diff, {'mask_loss_mode', 'name'})
        self.assertEqual(boundary['mask_loss_mode'], 'mabd_v1_boundary')
        self.assertEqual(boundary['mabd_v1_boundary_radius'], 1)
        self.assertFalse(boundary['use_cfsa'] or boundary['use_cfsav1'])
        self.assertEqual(boundary['samx_mode'], 'off')

    def test_empty_and_border_touching_masks_are_finite(self):
        gt, coeff, proto, boxes = self.sample()
        for filled in (False, True):
            gt.zero_()
            if filled:
                gt[:, :2, :2] = 1
            values, stats = mabd_v1_loss(gt, coeff, proto, torch.tensor([[0., 0., 2., 2.]]))
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue(all(torch.isfinite(v).all() for v in stats.values()))
            self.assertEqual(stats['empty_gt_masks'].item(), int(not filled))

    def test_duplicate_positive_samples_preserve_positive_average(self):
        gt, coeff, proto, boxes = self.sample()
        one, _ = mabd_v1_loss(gt, coeff, proto, boxes)
        two, _ = mabd_v1_loss(gt.repeat(2, 1, 1), coeff.repeat(2, 1), proto, boxes.repeat(2, 1))
        torch.testing.assert_close(one.mean(), two.mean())

    def test_config_dispatch_checkpoint_args_and_criterion_serialization(self):
        for mode in ('mabd_v1', 'mabd_v1_weights', 'mabd_v1_boundary'):
            criterion = SegmentationLoss(DummyModel(mode))
            gt, coeff, proto, boxes = self.sample()
            factor = None
            if mode == 'mabd_v1_boundary':
                factor, _ = criterion.morphology_factors(gt, boxes, torch.tensor([16.]))
            expected, _ = mabd_v1_loss(gt, coeff, proto, boxes, expand_boundary_roi=mode == 'mabd_v1',
                                       dice_shape_factor=factor)
            torch.testing.assert_close(criterion.single_mask_loss(gt, coeff, proto, boxes), expected.sum())
            self.assertEqual(criterion.mabd_v1_calls, 1)
            args = Model._reset_checkpoint_args({'mask_loss_mode': mode, 'mabd_v1_boundary_radius': 1})
            self.assertEqual(args, {'mask_loss_mode': mode, 'mabd_v1_boundary_radius': 1})
            buffer = io.BytesIO()
            torch.save(criterion, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, weights_only=False)
            self.assertEqual(loaded.mask_loss_mode, mode)

    def test_invalid_config_is_rejected(self):
        for options in ({'mask_loss_mode': 'invalid'}, {'mabd_v1_boundary_radius': 0},
                        {'mabd_v1_boundary_radius': True}, {'mabd_v1_boundary_radius': 1.5}):
            with self.assertRaises((TypeError, ValueError)):
                get_config(model_configuration=options)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_amp_large_reductions_are_fp32_and_gradients_finite(self):
        gt = torch.ones(1, 400, 400, device='cuda')
        proto = torch.zeros(1, 400, 400, device='cuda', requires_grad=True)
        coeff = torch.ones(1, 1, device='cuda', requires_grad=True)
        boxes = torch.tensor([[0., 0., 400., 400.]], device='cuda')
        for expanded, original_factor in ((False, None), (True, None), (False, torch.tensor([0.], device='cuda'))):
            proto.grad = None
            coeff.grad = None
            with torch.autocast('cuda', dtype=torch.float16):
                values, stats = mabd_v1_loss(gt, coeff, proto, boxes, expand_boundary_roi=expanded,
                                           dice_shape_factor=original_factor)
            self.assertEqual(values.dtype, torch.float32)
            self.assertTrue(torch.isfinite(values).all())
            values.sum().backward()
            self.assertTrue(torch.isfinite(proto.grad).all())
            self.assertGreater(proto.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
