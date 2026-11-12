import io
import os
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from samx_mask_experiment_config import MODES, options, select_offset_bound
from vajra.configs import get_config
from vajra.core.model import Model
from vajra.core.trainer import Trainer
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.modules import LDConv
from vajra.nn.samx_geometry import GeometricDifference, ResidualScale
from vajra.nn.samx_mask_variants import BoundedGeometricDifference, CoupledProtoSegmentation


class FixedOffset(nn.Module):
    def __init__(self, fields):
        super().__init__()
        self.weight = nn.Parameter(fields)

    def forward(self, x):
        return self.weight


class MaskVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def build(self, mode=None):
        args = options(mode or MODES[0], bound=2.)
        if mode is None:
            args.update(samx_proto_share_coeff=False, samx_proto_offset_bound=0.)
        cfg = get_config(model_configuration=args)
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = cfg, {'nc': 4}
        torch.manual_seed(42)
        model = trainer.get_model('vajra-v1-nano-seg', verbose=False)
        model.args = cfg
        return model, cfg

    @staticmethod
    def sample(device='cpu'):
        masks = torch.zeros(2, 16, 16, device=device)
        masks[:, 4:12, 4:12] = 1
        return dict(img=torch.rand(2, 3, 64, 64, device=device), masks=masks,
                    batch_idx=torch.tensor([0., 1.], device=device),
                    cls=torch.zeros(2, 1, device=device),
                    bboxes=torch.tensor([[.5, .5, .5, .5]] * 2, device=device))

    def test_bound_range_and_unit_initial_derivative(self):
        module = BoundedGeometricDifference(2, bound=2.)
        fields = torch.zeros(1, 6, 3, 5, requires_grad=True)
        actual = module._get_p(fields, fields.type())
        reference = module._get_p_0(3, 5, 3, fields.type()) + module.p_n
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        gradient, = torch.autograd.grad(actual.sum(), fields)
        torch.testing.assert_close(gradient, torch.ones_like(fields), atol=0, rtol=0)
        extreme = torch.linspace(-100, 100, fields.numel()).reshape_as(fields)
        displacement = module._get_p(extreme, extreme.type()) - reference
        self.assertLessEqual(displacement.abs().max().item(), 2.)
        self.assertTrue(torch.isfinite(displacement).all())

    def test_predeclared_bound_selection(self):
        for p95, expected in ((0., 1.), (.9, 1.), (1., 1.), (1.1, 2.), (2.1, 4.), (9., 16.)):
            self.assertEqual(select_offset_bound(p95), expected)
        for value in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                select_offset_bound(value)

    def test_bounded_sampling_against_independent_reference(self):
        torch.manual_seed(8)
        b, c, h, w, n = 2, 2, 5, 7, 3
        x = torch.randn(b, c, h, w, requires_grad=True)
        raw = torch.randn(b, 2*n, h, w) * .4
        module = BoundedGeometricDifference(c, bound=1.5)
        module.p_conv = FixedOffset(raw.clone())
        actual = module(x)
        xr = x.detach().clone().requires_grad_()
        pr = raw.clone().requires_grad_()
        kr = module.aggregate.weight.detach().clone().requires_grad_()
        q = module.p_n.reshape(1, 2, n, 1, 1).float()
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dy, dx = (1.5 * (pr / 1.5).tanh()).chunk(2, 1)
        fixed_y = (yy[None, None] + q[:, 0]).expand(b, -1, -1, -1)
        fixed_x = (xx[None, None] + q[:, 1]).expand(b, -1, -1, -1)
        def sample(y, z):
            grid = torch.stack((2*z/(w-1)-1, 2*y/(h-1)-1), -1)
            grid = grid.permute(0, 2, 3, 1, 4).reshape(b, h, w*n, 2)
            return F.grid_sample(xr, grid, padding_mode='border', align_corners=True).reshape(b, c, h, w, n)
        diff = sample(fixed_y+dy, fixed_x+dx) - sample(fixed_y, fixed_x)
        expected = F.conv2d(diff.permute(0, 1, 2, 4, 3).reshape(b, c, h*n, w), kr, stride=(n, 1))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        upstream = torch.randn_like(actual)
        (actual*upstream).sum().backward()
        (expected*upstream).sum().backward()
        for left, right in ((x.grad, xr.grad), (module.p_conv.weight.grad, pr.grad),
                            (module.aggregate.weight.grad, kr.grad)):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=3e-5)

    def test_constants_singletons_and_live_offset_gradients(self):
        for h, w in ((1, 1), (1, 7), (5, 1), (5, 7)):
            module = BoundedGeometricDifference(2)
            with torch.no_grad():
                module.p_conv.bias.fill_(.8)
            self.assertEqual(module(torch.full((2, 2, h, w), 3.7)).count_nonzero().item(), 0)
        module = BoundedGeometricDifference(2)
        module(torch.randn(2, 2, 5, 7)).square().sum().backward()
        # Squared zero residual has zero upstream gradient. A nonzero linear
        # objective verifies that zero offset does not create a dead branch.
        module.zero_grad()
        module(torch.randn(2, 2, 5, 7)).sum().backward()
        self.assertGreater(module.p_conv.weight.grad.abs().sum().item(), 0)

    def test_common_initialization_rng_and_identity_outputs(self):
        base, _ = self.build()
        base.eval()
        after_base = torch.rand(4)
        x = torch.rand(2, 3, 96, 96)
        with torch.inference_mode():
            reference = base(x)
        for mode in MODES:
            model, _ = self.build(mode)
            torch.testing.assert_close(torch.rand(4), after_base, atol=0, rtol=0)
            self.assertEqual(model.state_dict().keys(), base.state_dict().keys())
            for key, value in base.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
            model.eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(x), reference, atol=0, rtol=0)

    def test_coupling_changes_only_p3_coefficients_and_computes_once(self):
        base, _ = self.build()
        shared, _ = self.build(MODES[0])
        base.eval(); shared.eval()
        for model in (base, shared):
            model.model[-1].proto_enhance.geometry.p_conv.bias.data.fill_(.3)
        self.assertIs(type(shared.model[-1]), CoupledProtoSegmentation)
        calls = []
        handle = shared.model[-1].proto_enhance.register_forward_hook(lambda *args: calls.append(1))
        x = torch.rand(2, 3, 96, 96)
        with torch.inference_mode():
            old, new = base(x), shared(x)
        handle.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(new[1][0], old[1][0], atol=0, rtol=0)
        torch.testing.assert_close(new[1][2], old[1][2], atol=0, rtol=0)
        n3 = old[1][0][0].shape[-2] * old[1][0][0].shape[-1]
        self.assertGreater((new[1][1][..., :n3] - old[1][1][..., :n3]).abs().max().item(), 0)
        torch.testing.assert_close(new[1][1][..., n3:], old[1][1][..., n3:], atol=0, rtol=0)
        with torch.inference_mode():
            shared.model[-1].export = True
            shared.model[-1].format = 'onnx'
            exported = shared(x)
        torch.testing.assert_close(exported[0], new[0], atol=0, rtol=0)

    def test_loss_backward_optimizer_and_pickle(self):
        for mode in MODES:
            model, cfg = self.build(mode)
            batch = self.sample()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertFalse(model.criterion.use_morph_mask_loss)
            trainer = Trainer.__new__(Trainer); trainer.args = cfg
            optimizer = trainer.build_optimizer(model, name='SGD', lr=.01, decay=.0005)
            ids = [id(p) for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            expected = {id(p) for m in model.modules() if isinstance(m, LDConv) for p in m.p_conv.parameters()}
            actual = {id(p) for g in optimizer.param_groups if g.get('lr_scale') == .1 for p in g['params']}
            self.assertEqual(expected, actual)
            geometry = model.model[-1].proto_enhance.geometry
            self.assertGreater(geometry.p_conv.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(geometry.p_conv.weight.grad).all())
            if mode == MODES[1]:
                self.assertIs(type(geometry), BoundedGeometricDifference)
            stream = io.BytesIO(); torch.save(model.eval(), stream); stream.seek(0)
            reloaded = torch.load(stream, weights_only=False).eval()
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(batch['img']), model(batch['img']), atol=0, rtol=0)
            for name in ('samx_proto_share_coeff', 'samx_proto_offset_bound'):
                self.assertEqual(Model._reset_checkpoint_args(vars(cfg))[name], getattr(cfg, name))

    def test_invalid_configuration(self):
        for change in ({'samx_proto_offset_bound': -1}, {'samx_proto_offset_bound': float('nan')},
                       {'samx_proto_offset_bound': float('inf')}, {'samx_proto_offset_bound': True},
                       {'samx_proto_share_coeff': 1}, {'samx_proto_share_coeff': True},
                       {'samx_proto_mode': 'structured', 'samx_proto_offset_bound': 2.}):
            with self.assertRaises((ValueError, TypeError)):
                get_config(model_configuration=change)

    @unittest.skipUnless(os.environ.get('SAMX_TEST_CUDA') == '1' and torch.cuda.is_available(),
                         'GPU checks run after preceding training finishes')
    def test_amp_and_half_inference(self):
        for mode in MODES:
            model, _ = self.build(mode)
            model.cuda().train()
            batch = self.sample('cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            grad = model.model[-1].proto_enhance.geometry.p_conv.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            model.half().eval()
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(model(batch['img'].half())[0]).all())


if __name__ == '__main__':
    unittest.main()
