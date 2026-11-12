"""Checks for region coverage, thin-mask retention and observation-only hooks."""
import unittest
import tempfile
from pathlib import Path

import torch

from diagnose_cfsav1_regions import (RegionCollector, RegionGateObserver, make_regions,
                                    save_report, valid_mask, weighted_region_means)
from vajra.nn.cfsav1 import CFSAv1


class RegionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_regions_padding_and_class_background(self):
        valid = torch.ones(1, 1, 11, 15, dtype=torch.bool)
        valid[:, :, :2] = False
        fg = torch.zeros_like(valid)
        fg[:, :, 3:10, 2:9] = True
        other = torch.zeros_like(valid)
        other[:, :, 4:6, 12:14] = True
        regions = make_regions(fg, valid, 1, fg | other)
        self.assertFalse(regions[:, :, :2].any())
        self.assertEqual(int(regions[:, 1].sum()), 25)
        torch.testing.assert_close(regions[:, 1] | regions[:, 2], regions[:, 0])
        self.assertFalse((regions[:, 1] & regions[:, 2]).any())
        self.assertFalse((regions[:, 3:6] & other).any())
        self.assertFalse((regions[:, 4] & regions[:, 5]).any())

    def test_thin_coverage_means_and_empty_interiors(self):
        fg = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        fg[:, :, :, 3] = True
        regions = make_regions(fg, torch.ones_like(fg), 1)
        values = torch.arange(16.).reshape(1, 1, 4, 4)
        means, mass = weighted_region_means(values, regions)
        self.assertEqual(float(mass[0, 0]), 2.)
        self.assertTrue(torch.isnan(means[0, 1]).all())
        self.assertEqual(float(mass[0, 1]), 0.)
        self.assertEqual(float(means[0, 0, 0]), 7.)
        batch = {'img': torch.zeros(1, 3, 32, 40), 'ori_shape': [(16, 24)],
                 'ratio_pad': [((1., 1.), (8, 8))]}
        valid = valid_mask(batch, (8, 10))
        self.assertEqual(int(valid.sum()), 24)
        self.assertTrue(valid[:, :, 2:6, 2:8].all())

    def test_observer_does_not_change_forward(self):
        torch.manual_seed(42)
        for smooth in (False, True):
            model = CFSAv1(8, global_context=True, gate_smooth=smooth).eval()
            torch.nn.init.normal_(model.gate[-1].weight)
            x = torch.randn(2, 8, 7, 11)
            with torch.inference_mode():
                original = model(x)
                bands = model.bands(x).reshape(2, 4, 2, 3, 7, 11)
                with RegionGateObserver(model) as observer:
                    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
                    values = observer.latest
                    self.assertEqual(values.shape, (2, 18, 7, 11))
                    gains = values[:, :12].reshape(2, 4, 1, 3, 7, 11)
                    rebuilt = x + ((gains-1)*bands).sum(3).reshape_as(x)
                    torch.testing.assert_close(rebuilt, original, rtol=1e-5, atol=1e-6)
                    observer.latest = None
                self.assertFalse(observer.handles)
                torch.testing.assert_close(model(x), original, rtol=0, atol=0)

    def test_report_uses_measured_contrast_for_each_model(self):
        collector = RegionCollector(4)
        means = torch.ones(1, 6, 18)
        means[:, 0, :12] = 1.02
        collector.add('all', 2, ['sample.png'], means, torch.ones(1, 6))
        result = {'images':1, 'regions':collector.summary(), 'names':{}, 'checkpoint_sha256':'test'}
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder)/'report.md'
            save_report(report, result)
            content = report.read_text(encoding='utf-8')
        self.assertIn('0.01999998', content)
        self.assertNotIn('空间选择的依赖很弱', content)


if __name__ == '__main__':
    unittest.main()
