import copy
import unittest

import numpy as np
import torch

from test_mabd_v1 import DummyModel
from vajra.configs import get_config
from vajra.core.model import Model
from vajra.loss import SegmentationLoss
from vajra.loss.mabd_v1 import mabd_v1_loss
from mabd_followup_config import bounded_gain, candidate_passes, training_options


class FollowupPlanTests(unittest.TestCase):
    def test_e1_removes_only_point05_from_boundary_weight(self):
        gt=torch.zeros(1,10,12);gt[:,3:7,4:8]=1
        coeff=torch.ones(1,1);proto=torch.zeros(1,10,12,requires_grad=True)
        boxes=torch.tensor([[4.,3.,8.,7.]])
        original=DummyModel('mabd_v1_boundary')
        e1=DummyModel('mabd_v1_boundary')
        e1.args.mabd_v1_boundary_base=0.0;e1.args.mabd_v1_boundary_cap=.15
        c0,c1=SegmentationLoss(original),SegmentationLoss(e1)
        l0=c0.single_mask_loss(gt,coeff,proto,boxes);l1=c1.single_mask_loss(gt,coeff,proto,boxes)
        s0,s1=c0.mabd_v1_last_diagnostics,c1.mabd_v1_last_diagnostics
        for key in ('bce_mean','thinness_mean','dice_weight_mean','weighted_dice_mean','outer_boundary_pixels'):
            torch.testing.assert_close(s0[key],s1[key],rtol=0,atol=0)
        self.assertAlmostEqual(float(s0['boundary_weight_mean']-s1['boundary_weight_mean']),.05,places=6)
        self.assertAlmostEqual(float(l0-l1),.05*np.log(2),places=6)
        l1.backward();self.assertEqual(float(proto.grad[0,4,3]),0)

    def test_calibration_matches_mean_and_is_bounded(self):
        q=np.array([.02,.1,.5,1.])
        gain=bounded_gain(q,.09)
        weights=np.minimum(gain*q,.15)
        self.assertAlmostEqual(float(weights.mean()),.09,places=10)
        self.assertLessEqual(float(weights.max()),.15)
        self.assertGreaterEqual(float(weights.min()),0)
        self.assertEqual(gain,bounded_gain(q,.09))

    def test_unreachable_calibration_is_rejected(self):
        for q,target in (([0,0],.1),([],0),([np.nan],.1),([1.1],.1)):
            with self.assertRaises(ValueError):
                bounded_gain(q,target)

    def test_gain_is_capped_in_actual_loss(self):
        gt=torch.zeros(1,10,12);gt[:,3:7,4:8]=1
        loss,stats=mabd_v1_loss(gt,torch.ones(1,1),torch.zeros(1,10,12),torch.tensor([[4.,3.,8.,7.]]),
            expand_boundary_roi=False,dice_shape_factor=torch.zeros(1),boundary_base=0,boundary_gain=5,boundary_cap=.15)
        self.assertAlmostEqual(float(stats['boundary_weight_mean']),.15,places=6)
        self.assertTrue(torch.isfinite(loss).all())

    def test_invalid_hyperparameters_rejected(self):
        for key,value in (('mabd_v1_boundary_gain',float('nan')),('mabd_v1_boundary_gain',float('inf')),
                          ('mabd_v1_boundary_gain',True),('mabd_v1_boundary_base',-.1),
                          ('mabd_v1_boundary_cap',0),('mabd_v1_boundary_cap',2)):
            with self.assertRaises((ValueError,TypeError)):
                get_config(model_configuration={key:value})

    def test_checkpoint_and_experiment_options(self):
        options=training_options('E1')
        self.assertEqual(options['mabd_v1_boundary_base'],0)
        self.assertEqual(options['mabd_v1_boundary_gain'],.15)
        self.assertEqual(options['samx_mode'],'off');self.assertFalse(options['use_cfsav1'])
        keys=('mask_loss_mode','mabd_v1_boundary_base','mabd_v1_boundary_gain','mabd_v1_boundary_cap')
        expected={key:options[key] for key in keys}
        self.assertEqual(Model._reset_checkpoint_args(expected),expected)
        joint=training_options('joint',joint=True)
        self.assertTrue(joint['use_cfsav1']);self.assertEqual(joint['samx_mode'],'local_global')
        self.assertEqual(training_options('repeat',original=True)['mask_loss_mode'],'mabd')

    def test_screening_rejects_ap_gain_with_recall_collapse(self):
        reference=dict(metrics={key:.5 for key in ('metrics/mAP50(Mask)','metrics/mAP50-95(Mask)',
            'metrics/recall(Mask)','metrics/precision(Mask)')},boundary={'overall':{'f1':.5}},
            per_class={'0':{'instances':100,'map95':.5}})
        candidate=copy.deepcopy(reference)
        candidate['metrics']['metrics/mAP50-95(Mask)']+=.004
        self.assertTrue(candidate_passes(candidate,reference)['passed'])
        candidate['metrics']['metrics/recall(Mask)']-=.02
        self.assertFalse(candidate_passes(candidate,reference)['passed'])
        self.assertFalse(candidate_passes(reference,reference)['passed'])


if __name__=='__main__':
    unittest.main()
