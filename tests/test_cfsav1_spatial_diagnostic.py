"""Verify the diagnostic removes spatial variation after local/global fusion."""
import unittest
import torch
from torch.nn import functional as F
from vajra.nn.cfsav1 import CFSAv1
from diagnose_cfsav1_spatial import GateMapIntervention


class SpatialDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_post_tanh_mean_and_hook_restoration(self):
        torch.manual_seed(42)
        for smooth in [False, True]:
            m=CFSAv1(8, global_context=True, gate_smooth=smooth).eval()
            torch.nn.init.normal_(m.gate[-1].weight,std=2.)
            torch.nn.init.normal_(m.global_gate[-1].weight,std=2.)
            x=torch.randn(2,8,5,9)
            x[1] *= 3
            original=m(x)
            with torch.no_grad():
                bands=m.bands(x)
                from vajra.nn.cfsav1 import band_energy
                z=torch.cat((m.context(x),band_energy(bands,4).log1p()),1)
                logits=m.gate(z)+m.global_gate(z)
                d=.25*logits.tanh()
                if smooth:d=F.avg_pool2d(d,3,1,1,count_include_pad=False)
                mean=d.mean((-2,-1),keepdim=True).reshape(2,4,1,3,1,1)
                expected=x+(mean*bands.reshape(2,4,2,3,5,9)).sum(3).reshape_as(x)
                with GateMapIntervention(m,spatial_mean=True) as hooks:
                    torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
                    self.assertEqual(hooks.report()['image_module_evaluations'],2)
                    self.assertFalse(hooks.cache)
                self.assertFalse(hooks.handles)
                torch.testing.assert_close(m(x),original,rtol=0,atol=0)
                with GateMapIntervention(m,spatial_mean=False):
                    torch.testing.assert_close(m(x),original,rtol=0,atol=0)

    def test_identity_tiny_shapes_and_failure_cleanup(self):
        for h,w in [(1,1),(3,7)]:
            m=CFSAv1(8).eval()
            x=torch.randn(2,8,h,w)
            with torch.no_grad(),GateMapIntervention(m,True) as hooks:
                torch.testing.assert_close(m(x),x,rtol=0,atol=0)
                self.assertEqual(hooks.report()['spatial_std'],[0.,0.,0.])
            hooks=GateMapIntervention(m,True)
            with self.assertRaises(RuntimeError),hooks:
                raise RuntimeError('simulated validator failure')
            self.assertFalse(hooks.handles)
            torch.testing.assert_close(m(x),x,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
