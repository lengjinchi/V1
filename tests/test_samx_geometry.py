"""Numerical and integration checks for the geometric difference variant."""
import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from vajra.configs import get_config
from vajra.core.trainer import Trainer
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.modules import LDConv
from vajra.nn.samx_geometry import (GeometricDifference, GeometricResidualBlock,
                                    GeometryResidualSAMX, ResidualScale)


class FixedOffsets(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = nn.Parameter(values)

    def forward(self, x):
        return self.weight


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_offsets_give_exact_stable_path_and_live_offset_gradient(self):
        torch.manual_seed(13)
        block = GeometricResidualBlock(8, 8, rep_csp=True).eval()
        x = torch.randn(2, 8, 7, 11)
        with torch.no_grad():
            t = block.shared(x)
            self.assertEqual(block.geometry(t).count_nonzero().item(), 0)
            torch.testing.assert_close(block(x), block.regular_head(t), rtol=0, atol=0)
        block(x).square().mean().backward()
        self.assertGreater(block.geometry.p_conv.weight.grad.abs().sum().item(), 0)
        self.assertEqual(block.geometry.aggregate.weight.grad.count_nonzero().item(), 0)
        self.assertEqual(block.geometry_scale.weight.grad.item(), 0)
        with torch.no_grad():
            block.geometry.p_conv.weight.add_(-.1*block.geometry.p_conv.weight.grad)
        block.zero_grad(set_to_none=True)
        block(x).square().mean().backward()
        self.assertGreater(block.geometry.aggregate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(abs(block.geometry_scale.weight.grad.item()), 0)

    def test_constants_cancel_at_borders_and_singleton_dimensions(self):
        for h, w in ((1, 1), (1, 7), (5, 1), (5, 7)):
            m = GeometricDifference(2).eval()
            with torch.no_grad():
                m.p_conv.bias.copy_(torch.tensor([.2, -4., 8., -.4, 5., -8.]))
            self.assertEqual(m(torch.full((2, 2, h, w), 3.7)).count_nonzero().item(), 0)

    def test_output_and_gradients_match_independent_grid_sample_difference(self):
        torch.manual_seed(7)
        b, c, h, w, n = 2, 2, 5, 7, 3
        x = torch.randn(b, c, h, w, requires_grad=True)
        offsets = torch.randn(b, 2*n, h, w)*1.3
        m = GeometricDifference(c, n)
        m.p_conv = FixedOffsets(offsets.clone())
        actual = m(x)
        xr = x.detach().clone().requires_grad_()
        ofr = offsets.clone().requires_grad_()
        kr = m.aggregate.weight.detach().clone().requires_grad_()
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        base = torch.cat((yy[None].expand(n,-1,-1), xx[None].expand(n,-1,-1)))[None]
        def sample(delta):
            p = base+m.p_n+delta
            gy, gx = p[:,:n].permute(0,2,3,1), p[:,n:].permute(0,2,3,1)
            grid = torch.stack((2*gx/(w-1)-1, 2*gy/(h-1)-1),-1).reshape(b,h,w*n,2)
            return F.grid_sample(xr, grid, padding_mode='border', align_corners=True).reshape(b,c,h,w,n)
        difference = sample(ofr)-sample(torch.zeros_like(ofr))
        packed = difference.permute(0,1,2,4,3).reshape(b,c,h*n,w)
        expected = F.conv2d(packed, kr, stride=(n,1))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        upstream = torch.randn_like(actual)
        (actual*upstream).sum().backward()
        (expected*upstream).sum().backward()
        for a, e in ((x.grad,xr.grad),(m.p_conv.weight.grad,ofr.grad),(m.aggregate.weight.grad,kr.grad)):
            torch.testing.assert_close(a,e,atol=1e-5,rtol=2e-5)

    def test_optimizer_includes_scales_and_slow_offsets_once(self):
        model = GeometryResidualSAMX(16,16,16,8,rep_csp=True)
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(samx_offset_lr_scale=.1)
        optimizer = trainer.build_optimizer(model, name='SGD', lr=.01, decay=.0005)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        offsets = {id(p) for m in model.modules() if isinstance(m,LDConv) for p in m.p_conv.parameters()}
        self.assertEqual(len(offsets),4)
        self.assertEqual({id(p) for g in optimizer.param_groups if g.get('lr_scale')==.1 for p in g['params']},offsets)
        scales = [m.weight for m in model.modules() if isinstance(m,ResidualScale)]
        self.assertEqual(len(scales),2)
        for p in scales:
            self.assertEqual(ids.count(id(p)),1)
        for g in optimizer.param_groups:
            if g.get('lr_scale')==.1:
                self.assertAlmostEqual(g['lr'],.001)

    def test_config_rejects_dead_branch_initial_gain(self):
        for gain in (0,-.1,1.1,True):
            with self.assertRaises((ValueError,TypeError)):
                get_config(model_configuration={'samx_mode':'geometry_residual','samx_geometry_gain':gain})
        self.assertEqual(get_config(model_configuration={'samx_mode':'geometry_residual'}).samx_geometry_gain,.1)

    def test_real_segmentation_loss_backward_and_pickle_roundtrip(self):
        args = get_config(model_configuration={'samx_mode':'geometry_residual','use_cfsa':False,'use_cfsav1':False})
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = args, {'nc':4}
        model = trainer.get_model('vajra-v1-nano-seg',verbose=False)
        self.assertIs(type(model.model[0].vajra_neck3),GeometryResidualSAMX)
        self.assertIsInstance(model.model[-1].proto_enhance,nn.Identity)
        self.assertIsInstance(model.model[-1].seg_enhance,nn.Identity)
        model.args = args
        sample={'img':torch.rand(2,3,64,64),'masks':torch.zeros(2,16,16),
                'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
                'bboxes':torch.tensor([[.5,.5,.5,.5]]*2)}
        sample['masks'][:,4:12,4:12]=1
        loss,items = model(sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.criterion.use_morph_mask_loss)
        loss.backward()
        for m in model.modules():
            if isinstance(m,GeometricDifference):
                self.assertTrue(torch.isfinite(m.p_conv.weight.grad).all())
                self.assertGreater(m.p_conv.weight.grad.abs().sum().item(),0)
        model.eval()
        with torch.inference_mode(): expected=model(sample['img'])
        stream=io.BytesIO();torch.save(model,stream);stream.seek(0)
        restored=torch.load(stream,weights_only=False).eval()
        with torch.inference_mode():torch.testing.assert_close(restored(sample['img']),expected,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_amp_training_and_half_inference(self):
        model=GeometryResidualSAMX(16,16,16,8,rep_csp=True).cuda()
        x=torch.randn(2,16,7,11,device='cuda',requires_grad=True)
        with torch.autocast('cuda',dtype=torch.float16):y=model(x)
        y.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for m in model.modules():
            if isinstance(m,GeometricDifference):
                self.assertTrue(torch.isfinite(m.p_conv.weight.grad).all())
                self.assertGreater(m.p_conv.weight.grad.abs().sum().item(),0)
        model.half().eval()
        with torch.inference_mode():self.assertTrue(torch.isfinite(model(x.half())).all())
        m=GeometricDifference(2).cuda().half().eval()
        with torch.no_grad():m.p_conv.bias.uniform_(-2,2)
        with torch.inference_mode():self.assertEqual(m(torch.full((2,2,5,7),3.7,device='cuda',dtype=torch.float16)).count_nonzero().item(),0)


if __name__=='__main__':unittest.main()
