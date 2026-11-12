"""Behavioral checks for repaired sampling and independent SAMX ablations."""
import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from vajra.configs import get_config
from vajra.core.model import Model
from vajra.core.trainer import Trainer
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.modules import LDConv, ShapeAdaptiveMerudandaX
from vajra.nn.samx import LocalGlobalSAMX, StageGlobalSAMX


class FixedOffsets(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(value)

    def forward(self, x):
        return self.value


class SAMXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_offset_initialization(self):
        module = LDConv(2, 2, 3)
        self.assertEqual(module.p_conv.weight.count_nonzero().item(), 0)
        self.assertEqual(module.p_conv.bias.count_nonzero().item(), 0)
        self.assertFalse(module.p_conv._backward_hooks)

    def test_constant_preservation_including_borders_and_singletons(self):
        for h, w in ((1, 1), (1, 7), (5, 1), (5, 7)):
            module = LDConv(2, 2, 3).eval()
            module.conv = nn.Identity()
            with torch.no_grad():
                module.p_conv.bias.copy_(torch.tensor([0.2, -4., 8., -0.4, 5., -8.]))
            torch.testing.assert_close(module(torch.ones(2, 2, h, w)),
                                       torch.ones(2, 2, h * 3, w))

    def test_sampler_and_gradients_match_grid_sample_border_reference(self):
        torch.manual_seed(3)
        b, c, h, w, n = 2, 2, 4, 7, 3
        x = torch.randn(b, c, h, w, requires_grad=True)
        offsets = torch.randn(b, 2*n, h, w) * 2.4
        module = LDConv(c, c, n).eval()
        module.p_conv, module.conv = FixedOffsets(offsets), nn.Identity()
        output = module(x).reshape(b, c, h, n, w).permute(0, 1, 2, 4, 3)
        xr = x.detach().clone().requires_grad_(True)
        ofr = offsets.detach().clone().requires_grad_(True)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        base = torch.cat((yy[None].expand(n, -1, -1), xx[None].expand(n, -1, -1)))[None]
        coords = base + module.p_n + ofr
        gy, gx = coords[:, :n].permute(0,2,3,1), coords[:, n:].permute(0,2,3,1)
        grid = torch.stack((2*gx/(w-1)-1, 2*gy/(h-1)-1), -1).reshape(b,h,w*n,2)
        reference = F.grid_sample(xr, grid, padding_mode="border", align_corners=True).reshape(b,c,h,w,n)
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
        weights = torch.randn_like(output)
        (output*weights).sum().backward()
        (reference*weights).sum().backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(module.p_conv.value.grad, ofr.grad, atol=2e-5, rtol=2e-5)

    def test_optimizer_offset_membership_decay_and_scheduled_lr_ratio(self):
        model = nn.Sequential(nn.Conv2d(2,2,1), LDConv(2,2,3))
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(samx_offset_lr_scale=.1)
        optimizer = trainer.build_optimizer(model, name="SGD", lr=.01, decay=.0005)
        all_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
        offsets = {id(p) for p in model[1].p_conv.parameters()}
        slow_ids = {id(p) for g in optimizer.param_groups if g.get('lr_scale') == .1 for p in g['params']}
        self.assertEqual(slow_ids, offsets)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1-.09*e)
        for _ in range(3):
            for group in optimizer.param_groups:
                ratio = .1 if any(id(p) in offsets for p in group['params']) else 1.
                self.assertAlmostEqual(group['lr'], optimizer.param_groups[0]['lr'] * ratio)
            optimizer.step(); scheduler.step()
        slow = [g for g in optimizer.param_groups if g.get('lr_scale') == .1]
        self.assertEqual(len(slow), 2)
        self.assertEqual(sorted(g['weight_decay'] for g in slow), [0., .0005])
        self.assertEqual(sorted(g['warmup_bias'] for g in slow), [False, True])

    def test_local_global_recovers_original_at_init_then_learns_spatial_weights(self):
        torch.manual_seed(42)
        original = ShapeAdaptiveMerudandaX(16,16,16,8,rep_csp=True).eval()
        torch.manual_seed(42)
        module = LocalGlobalSAMX(16,16,16,8,rep_csp=True).eval()
        x = torch.randn(2,16,7,11)
        torch.testing.assert_close(module(x), original(x), rtol=0, atol=0)
        module(x).square().mean().backward()
        self.assertGreater(module.local_route[-1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.local_route[-1].weight.add_(-.1*module.local_route[-1].weight.grad)
        ac, al = module._get_route_weight(x)
        self.assertEqual(tuple(ac.shape), (2,1,7,11))
        torch.testing.assert_close(ac+al, torch.ones_like(ac))
        self.assertGreater(al.std((-2,-1)).mean().item(), 0)
        together = module(x)
        torch.testing.assert_close(module(x[:1]), together[:1], atol=1e-6, rtol=1e-5)

    def test_stage_gates_are_independent_and_read_their_stage_inputs(self):
        module = StageGlobalSAMX(16,16,16,8,rep_csp=True).eval()
        inputs, outputs = {}, {}
        handles=[]
        for label in ('route2','route3'):
            def record(m, args, result, label=label):
                inputs[label], outputs[label] = args[0].detach(), result.detach()
            handles.append(getattr(module,label).register_forward_hook(record))
        x=torch.randn(2,16,7,11)
        before=module(x)
        self.assertEqual(tuple(outputs['route2'].shape),(2,2,1,1))
        self.assertFalse(torch.equal(inputs['route2'], inputs['route3']))
        with torch.no_grad(): module.route3[-1].bias.add_(torch.tensor([3.,-3.]))
        after=module(x)
        self.assertGreater((before-after).abs().max().item(),0)
        for handle in handles:handle.remove()

    def test_config_and_checkpoint_options(self):
        for mode in ('off','original','stage_global','local_global'):
            args=get_config(model_configuration={'samx_mode':mode})
            self.assertEqual(Model._reset_checkpoint_args(vars(args))['samx_mode'], mode)
        for changes in ({'samx_mode':'typo'},{'samx_offset_lr_scale':0}):
            with self.assertRaises(ValueError):get_config(model_configuration=changes)

    def test_real_model_routing_common_initialization_loss_and_pickle(self):
        originals = None
        for mode in ('original','stage_global','local_global'):
            args=get_config(model_configuration={'samx_mode':mode})
            trainer=SegmentationTrainer.__new__(SegmentationTrainer)
            trainer.args, trainer.data=args, {'nc':2}
            torch.manual_seed(42)
            model=trainer.get_model('vajra-v1-nano-seg',verbose=False)
            self.assertIsInstance(model.model[-1].proto_enhance,nn.Identity)
            self.assertFalse(model.model[-1].use_cfsav1)
            state=model.state_dict()
            if originals is None: originals={k:v.clone() for k,v in state.items()}
            else:
                for k,v in originals.items():
                    if k in state and state[k].shape==v.shape:
                        torch.testing.assert_close(state[k],v,rtol=0,atol=0)
            model.args=args
            sample={'img':torch.rand(2,3,64,64),'masks':torch.zeros(2,16,16),
                'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
                'bboxes':torch.tensor([[.5,.5,.5,.5]]*2)}
            sample['masks'][:,4:12,4:12]=1
            loss,items=model(sample)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertFalse(model.criterion.use_morph_mask_loss)
            for sub in model.modules():
                if isinstance(sub,LDConv):
                    self.assertTrue(torch.isfinite(sub.p_conv.weight.grad).all())
                    self.assertGreater(sub.p_conv.weight.grad.abs().sum().item(),0)
            model.eval()
            with torch.inference_mode(): expected=model(sample['img'])
            data=io.BytesIO();torch.save(model,data);data.seek(0)
            reloaded=torch.load(data,weights_only=False).eval()
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(sample['img']),expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_amp_backward_and_half_checkpoint_inference(self):
        for cls in (ShapeAdaptiveMerudandaX,StageGlobalSAMX,LocalGlobalSAMX):
            module=cls(16,16,16,8,rep_csp=True).cuda()
            x=torch.randn(2,16,7,11,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.float16): y=module(x)
            y.float().square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for sub in module.modules():
                if isinstance(sub,LDConv):self.assertTrue(torch.isfinite(sub.p_conv.weight.grad).all())
            module.half().eval()
            with torch.inference_mode():self.assertTrue(torch.isfinite(module(x.half())).all())


if __name__=='__main__':unittest.main()
