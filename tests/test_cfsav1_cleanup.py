"""Regressions for removing paired/reference training while reading old runs."""
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from vajra.configs import get_config
from vajra.nn.cfsav1 import CFSAv1
from vajra.models.vajra.segment.train import SegmentationTrainer


class CleanupTests(unittest.TestCase):
    def test_disabled_historical_options_are_not_saved(self):
        args = get_config(model_configuration={
            'use_cfsav1': True, 'cfsav1_paired': False, 'cfsav1_sensitivity': False,
            'cfsav1_reference': None, 'cfsav1_calibration_batches': 8,
            'cfsav1_tau_scale': .05, 'cfsav1_perturb_strength': 1., 'cfsav1_sensitivity_gain': .1,
        })
        self.assertTrue(args.use_cfsav1)
        self.assertNotIn('cfsav1_paired', vars(args))
        self.assertNotIn('cfsav1_sensitivity_gain', vars(args))
        self.assertNotIn('cfsav1_reference', vars(args))

    def test_removed_training_cannot_silently_resume(self):
        for key, value in [('cfsav1_paired', True), ('cfsav1_sensitivity', True),
                           ('cfsav1_reference', 'old-reference.pt')]:
            with self.assertRaisesRegex(ValueError, 'removed'):
                get_config(model_configuration={key: value})

    def test_training_entry_is_single_view_with_cfsav1_disabled(self):
        import seg_train
        model = MagicMock()
        with patch.object(seg_train, 'Vajra', return_value=model):
            seg_train.main()
        args = model.train.call_args.kwargs
        for word in ('paired', 'sensitivity', 'reference', 'calibration', 'perturb', 'tau'):
            self.assertFalse(any(word in k for k in args))
        self.assertNotIn('_paired', args['name'])
        self.assertNotIn('_sens', args['name'])
        self.assertFalse(args['use_cfsa'])
        self.assertFalse(args['use_cfsav1'])
        self.assertFalse(args['cfsav1_global_context'])
        self.assertFalse(args['cfsav1_semantic_context'])
        self.assertEqual(args['samx_mode'], 'geometry_residual')
        self.assertEqual(args['cfsav1_gate_ramp_epochs'], 0)
        self.assertFalse(args['cfsav1_gate_smooth'])
        get_config(model_configuration=args)

    def test_preprocess_does_not_duplicate_images_or_targets(self):
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args = get_config(model_configuration={'use_cfsav1': False})
        trainer.epoch = 0
        trainer.device = torch.device('cpu')
        trainer.stride = 32
        batch = {'img': torch.full((2, 3, 64, 64), 128, dtype=torch.uint8),
                 'cls': torch.tensor([[0.], [1.]]), 'batch_idx': torch.tensor([0., 1.]),
                 'masks': torch.ones(2, 16, 16)}
        out = trainer.preprocess_batch(batch)
        self.assertEqual(out['img'].shape[0], 2)
        self.assertEqual(out['masks'].shape[0], 2)
        self.assertEqual(out['cls'].shape[0], 2)
        self.assertFalse(any(k.startswith('cfsav1_') for k in out))

    def test_legacy_sensitivity_pickle_inference_only(self):
        # Recreate old pickle fields without a constructor for the retired head.
        m = CFSAv1(8)
        m.use_sensitivity = True
        m.sensitivity_head = nn.Sequential(nn.Conv2d(28, 16, 1), nn.SiLU(), nn.Conv2d(16, 12, 1))
        m.gate = nn.Sequential(nn.Conv2d(40, 16, 1), nn.SiLU(), nn.Conv2d(16, 12, 1))
        x = torch.rand(2, 8, 5, 7)
        with self.assertRaisesRegex(RuntimeError, 'removed'):
            m(x)
        m.eval()
        with torch.no_grad():
            bands = m.bands(x)
            from vajra.nn.cfsav1 import band_energy
            context = m.context(x)
            z = torch.cat((context, band_energy(bands, 4).log1p()), 1)
            z = torch.cat((z, m.sensitivity_head(z).sigmoid()), 1)
            delta = (.25*m.gate(z).tanh()).reshape(2, 4, 1, 3, 5, 7)
            expected = x+(delta*bands.reshape(2, 4, 2, 3, 5, 7)).sum(3).reshape_as(x)
            torch.testing.assert_close(m(x), expected)


if __name__ == '__main__':
    unittest.main()
