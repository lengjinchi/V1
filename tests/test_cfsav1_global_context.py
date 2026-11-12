"""Check global context influence, bounded gains and old checkpoint compatibility."""
import copy
import io
import unittest

import torch
from torch import nn

from vajra.configs import get_config
from vajra.core.model import Model
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.cfsav1 import CFSAv1


class GlobalContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_gradient_and_small_rectangular_shapes(self):
        torch.manual_seed(42)
        for smooth in (False, True):
            for h, w in [(1, 1), (5, 17)]:
                m = CFSAv1(8, global_context=True, gate_smooth=smooth)
                x = torch.randn(2, 8, h, w, requires_grad=True)
                y = m(x)
                torch.testing.assert_close(y, x, rtol=0, atol=0)
                y.square().mean().backward()
                self.assertGreater(m.global_gate[-1].weight.grad.abs().sum().item(), 0)
                self.assertGreater(m.gate[-1].weight.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_remote_background_changes_local_output_without_batch_mixing(self):
        # Left pixel and its local receptive field stay identical; only distant
        # right-hand background changes. Energy supplies a deterministic signal.
        m = CFSAv1(8, global_context=True).eval()
        with torch.no_grad():
            for p in m.global_gate.parameters():
                p.zero_()
            m.global_gate[1].weight[:, m.context_channels:, 0, 0].fill_(.1)
            m.global_gate[-1].weight.fill_(.1)
        a = torch.ones(1, 8, 9, 31)
        b = a.clone()
        b[..., 20:] = 4
        ya, yb = m(a), m(b)
        self.assertGreater((ya[..., :3] - yb[..., :3]).abs().max().item(), 1e-4)
        together = m(torch.cat([a, b]))
        torch.testing.assert_close(together[:1], ya)
        torch.testing.assert_close(together[1:], yb)
        m.global_gate = None
        torch.testing.assert_close(m(a)[..., :3], m(b)[..., :3], rtol=0, atol=0)

    def test_combined_gain_bound_and_serialization(self):
        m = CFSAv1(8, rho=.25, global_context=True)
        with torch.no_grad():
            m.gate[-1].bias.fill_(100)
            m.global_gate[-1].bias.fill_(100)
        x = torch.randn(2, 8, 7, 15)
        # Both branches saturate, but rho is applied only once.
        torch.testing.assert_close(m(x), 1.25*x)
        data = io.BytesIO()
        torch.save(m, data)
        data.seek(0)
        restored = torch.load(data, weights_only=False)
        self.assertTrue(restored.global_context)
        torch.testing.assert_close(restored(x), m(x), rtol=0, atol=0)
        clone = CFSAv1(8, global_context=True)
        clone.load_state_dict(m.state_dict(), strict=True)
        torch.testing.assert_close(clone(x), m(x), rtol=0, atol=0)

    def test_disabled_branch_preserves_old_pickle_and_state(self):
        m = CFSAv1(8)
        nn.init.normal_(m.gate[-1].weight, std=.1)
        old = copy.deepcopy(m)
        del old.global_gate
        del old.global_context
        x = torch.randn(2, 8, 9, 13)
        torch.testing.assert_close(m(x), old(x), rtol=0, atol=0)
        self.assertFalse(any('global' in k for k in m.state_dict()))
        m.load_state_dict(old.state_dict(), strict=True)

    def test_trainer_options_weight_transfer_and_trace(self):
        args = get_config(model_configuration={'use_cfsav1': True, 'cfsav1_global_context': True})
        self.assertTrue(Model._reset_checkpoint_args(vars(args))['cfsav1_global_context'])
        with self.assertRaises(TypeError):
            get_config(model_configuration={'cfsav1_global_context': 'True'})
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = args, {'nc': 2}
        m = trainer.get_model('vajra-v1-nano-seg', verbose=False)
        gate = m.model[-1].proto_enhance
        self.assertTrue(gate.global_context)
        with torch.no_grad():
            gate.global_gate[-1].weight.fill_(.02)
        restored = trainer.get_model('checkpoint.pt', weights=m, verbose=False)
        torch.testing.assert_close(restored.model[-1].proto_enhance.global_gate[-1].weight,
                                   gate.global_gate[-1].weight)
        small = CFSAv1(8, global_context=True).eval()
        nn.init.normal_(small.global_gate[-1].weight, std=.1)
        traced = torch.jit.trace(small, torch.randn(1, 8, 9, 11))
        x = torch.randn(2, 8, 5, 17)
        torch.testing.assert_close(traced(x), small(x))


if __name__ == '__main__':
    unittest.main()
