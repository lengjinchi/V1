"""P4 hints: identity, independent samples, Proto-only routing and saved options."""
import copy
import io
import unittest

import torch
from torch.nn import functional as F

from diagnose_cfsav1_spatial import GateMapIntervention
from diagnose_cfsav1_regions import RegionGateObserver
from vajra.configs import get_config
from vajra.core.model import Model
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.cfsav1 import CFSAv1
from vajra.nn.head import Segmentation


class SemanticContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_and_semantic_learning(self):
        torch.manual_seed(42)
        for h, w in ((1, 1), (7, 13)):
            module = CFSAv1(8, global_context=True, semantic_channels=16)
            x = torch.randn(2, 8, h, w, requires_grad=True)
            p4 = torch.randn(2, 16, max(1, h//2), max(1, w//2), requires_grad=True)
            y = module(x, p4)
            torch.testing.assert_close(y, x, rtol=0, atol=0)
            y.square().mean().backward()
            self.assertGreater(module.semantic_gate[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(x.grad).all())
            # The zero last layer delays gradients to earlier P4 layers at init.
            with torch.no_grad():
                module.semantic_gate[-1].weight.add_(-.01 * module.semantic_gate[-1].weight.grad)
            module.zero_grad(set_to_none=True)
            module(x, p4).square().mean().backward()
            self.assertGreater(p4.grad.abs().sum().item(), 0)
            self.assertGreater(module.semantic_gate[0].weight.grad.abs().sum().item(), 0)

    def test_semantic_influence_without_batch_mixing_and_bounded_gain(self):
        torch.manual_seed(42)
        module = CFSAv1(8, global_context=True, semantic_channels=16).eval()
        torch.nn.init.normal_(module.semantic_gate[-1].weight, std=.2)
        x = torch.rand(1, 8, 9, 17)
        a, b = torch.randn(1, 16, 5, 9), torch.randn(1, 16, 5, 9) * 3
        ya, yb = module(x, a), module(x, b)
        self.assertGreater((ya-yb).abs().max().item(), 1e-5)
        together = module(torch.cat((x, x)), torch.cat((a, b)))
        torch.testing.assert_close(together[:1], ya)
        torch.testing.assert_close(together[1:], yb)
        with torch.no_grad():
            for gate in (module.gate, module.global_gate, module.semantic_gate):
                gate[-1].weight.zero_()
                gate[-1].bias.fill_(100)
        torch.testing.assert_close(module(x, a), 1.25*x)
        with self.assertRaisesRegex(ValueError, 'no P4'):
            module(x)
        with self.assertRaisesRegex(ValueError, 'semantic channels'):
            module(x, torch.rand(1, 8, 5, 9))

    def test_proto_only_routing_and_legacy_inference(self):
        torch.manual_seed(42)
        original = Segmentation(2, in_channels=[8, 16, 32], use_cfsav1=True,
                                cfsav1_global_context=True).train()
        guided = copy.deepcopy(original)
        guided.configure_enhancement(use_cfsav1=True, global_context=True, semantic_context=True)
        missing, unexpected = guided.load_state_dict(original.state_dict(), strict=False)
        self.assertTrue(missing and all('semantic_gate' in k for k in missing))
        self.assertFalse(unexpected)
        features = [torch.randn(2, 8, 9, 17), torch.randn(2, 16, 5, 9), torch.randn(2, 32, 3, 5)]
        torch.testing.assert_close(guided(features), original(features), rtol=0, atol=0)
        torch.nn.init.normal_(guided.proto_enhance.semantic_gate[-1].weight, std=.3)
        baseline, changed = original(features), guided(features)
        torch.testing.assert_close(changed[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(changed[1], baseline[1], rtol=0, atol=0)
        self.assertGreater((changed[2]-baseline[2]).abs().max().item(), 1e-6)
        # Full-module pickles predating this feature contain neither new field.
        old = copy.deepcopy(original.proto_enhance).eval()
        del old.semantic_gate
        del old.semantic_channels
        x = features[0]
        torch.testing.assert_close(old(x), original.proto_enhance(x), rtol=0, atol=0)

    def test_options_trainer_restore_and_full_loss_backward(self):
        config = {'use_cfsav1': True, 'cfsav1_global_context': True,
                  'cfsav1_semantic_context': True}
        args = get_config(model_configuration=config)
        self.assertTrue(Model._reset_checkpoint_args(vars(args))['cfsav1_semantic_context'])
        with self.assertRaises(TypeError):
            get_config(model_configuration={**config, 'cfsav1_semantic_context':'True'})
        with self.assertRaises(ValueError):
            get_config(model_configuration={'cfsav1_semantic_context': True})
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = args, {'nc': 2}
        model = trainer.get_model('vajra-v1-nano-seg', verbose=False)
        module = model.model[-1].proto_enhance
        self.assertEqual(module.semantic_channels, model.model[-1].in_channels[1])
        model.args = args
        masks = torch.zeros(2, 16, 16)
        masks[:, 4:12, 4:12] = 1
        sample = {'img':torch.rand(2,3,64,64), 'masks':masks,
            'batch_idx':torch.tensor([0.,1.]), 'cls':torch.zeros(2,1),
            'bboxes':torch.tensor([[.5,.5,.5,.5]]*2)}
        loss, items = model(sample)
        self.assertEqual(items.shape, (4,))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(module.semantic_gate[-1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.semantic_gate[-1].weight.fill_(.02)
        restored = trainer.get_model('checkpoint.pt', weights=model, verbose=False)
        torch.testing.assert_close(restored.model[-1].proto_enhance.semantic_gate[-1].weight,
                                   module.semantic_gate[-1].weight)
        model.eval()
        with torch.inference_mode():
            expected = model(sample['img'])
        data = io.BytesIO()
        torch.save(model, data)
        data.seek(0)
        reloaded = torch.load(data, weights_only=False).eval()
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(sample['img']), expected, rtol=0, atol=0)

    def test_diagnostics_include_p4_before_tanh_and_spatial_mean(self):
        torch.manual_seed(42)
        module = CFSAv1(8, global_context=True, semantic_channels=16, gate_smooth=True).eval()
        torch.nn.init.normal_(module.semantic_gate[-1].weight, std=2.)
        x, p4 = torch.randn(2,8,7,11), torch.randn(2,16,4,6)
        with torch.inference_mode():
            original = module(x,p4)
            with RegionGateObserver(module) as observer:
                torch.testing.assert_close(module(x,p4), original, rtol=0, atol=0)
                gains = observer.latest[:,:12].reshape(2,4,1,3,7,11)
                observer.latest = None
            bands = module.bands(x).reshape(2,4,2,3,7,11)
            expected = x + ((gains-1).mean((-2,-1),keepdim=True)*bands).sum(3).reshape_as(x)
            with GateMapIntervention(module, spatial_mean=True):
                torch.testing.assert_close(module(x,p4), expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(module(x,p4), original, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_backward_and_half_inference(self):
        module = CFSAv1(8, global_context=True, semantic_channels=16).cuda()
        x = torch.rand(2,8,9,17,device='cuda',requires_grad=True)
        p4 = torch.rand(2,16,5,9,device='cuda',requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = module(x,p4).float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(module.semantic_gate[-1].weight.grad).all())
        self.assertGreater(module.semantic_gate[-1].weight.grad.abs().sum().item(), 0)
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(module.half()(x.half(),p4.half())).all())


if __name__ == '__main__':
    unittest.main()
