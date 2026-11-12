"""Regression checks for scheduled gates, legacy weights and actual trainer resume."""
import copy
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vajra.configs import get_config
from vajra.core.model import Model
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.cfsav1 import CFSAv1
from vajra.utils.torch_utils import ModelEMA


class GateRampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_residual_scaling_gradients_and_checkpoint_inference(self):
        module = CFSAv1(8)
        nn.init.normal_(module.gate[-1].weight, std=.2)
        x = torch.randn(2, 8, 9, 11, requires_grad=True)
        full = module(x)
        module.gate_scale.fill_(.1)
        small = module(x)
        torch.testing.assert_close(small - x, (full - x) * .1, atol=3e-7, rtol=1e-4)
        small.square().mean().backward()
        self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        module.eval()
        torch.testing.assert_close(module(x), small)
        self.assertIn('gate_scale', dict(module.named_buffers()))
        self.assertNotIn('gate_scale', dict(module.named_parameters()))
        stream = io.BytesIO()
        torch.save(module, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        torch.testing.assert_close(restored(x), small)

    def test_legacy_pickle_and_strict_state_dict_restore_full_strength(self):
        original = CFSAv1(8)
        nn.init.normal_(original.gate[-1].weight, std=.2)
        legacy = copy.deepcopy(original)
        del legacy.gate_scale
        sample = torch.randn(2, 8, 7, 9)
        torch.testing.assert_close(legacy(sample), original(sample), rtol=0, atol=0)
        # Nested strict loading is how the trainer restores EMA on resume.
        current = nn.Sequential(CFSAv1(8))
        current[0].gate_scale.fill_(.1)
        current.load_state_dict(nn.Sequential(legacy).state_dict(), strict=True)
        self.assertEqual(current[0].gate_scale.item(), 1)
        torch.testing.assert_close(current(sample), original(sample), rtol=0, atol=0)

    def test_schedule_endpoints_disabled_and_exact_ema_sync(self):
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args = get_config(model_configuration=dict(use_cfsav1=True, cfsav1_gate_ramp_epochs=20))
        trainer.model = nn.Sequential(CFSAv1(8))
        trainer.ema = ModelEMA(trainer.model)
        for epoch, expected in [(0,.1),(10,.55),(20,1),(77,1)]:
            trainer.epoch = epoch
            trainer._update_cfsav1_gate_scale()
            self.assertAlmostEqual(trainer.model[0].gate_scale.item(), expected, places=6)
            self.assertAlmostEqual(trainer.ema.ema[0].gate_scale.item(), expected, places=6)
        trainer.epoch = 10
        trainer._update_cfsav1_gate_scale()
        # Deliberately stale EMA buffer: the post-optimizer synchronization must repair it.
        trainer.ema.ema[0].gate_scale.fill_(.1)
        trainer.ema.update(trainer.model)
        trainer._update_cfsav1_gate_scale(ema_only=True)
        torch.testing.assert_close(trainer.ema.ema[0].gate_scale, trainer.model[0].gate_scale, rtol=0, atol=0)
        trainer.args.cfsav1_gate_ramp_epochs = 0
        trainer._update_cfsav1_gate_scale()
        self.assertEqual(trainer.model[0].gate_scale.item(), 1)

    def test_config_and_public_checkpoint_preservation(self):
        for bad in [dict(cfsav1_gate_ramp_epochs=-1), dict(cfsav1_gate_ramp_epochs=True),
                    dict(cfsav1_gate_ramp_epochs=2.5), dict(cfsav1_gate_ramp_start=0),
                    dict(cfsav1_gate_ramp_start=float('nan')), dict(cfsav1_gate_ramp_start=True),
                    dict(cfsav1_gate_ramp_epochs=20, use_cfsav1=False)]:
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                get_config(model_configuration={'use_cfsav1': True, **bad})
        args = dict(use_cfsav1=True, cfsav1_gate_ramp_epochs=20, cfsav1_gate_ramp_start=.1)
        self.assertEqual(Model._reset_checkpoint_args(args), args)

    def test_schedule_after_inference_mode_validation_casts(self):
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args = get_config(model_configuration=dict(use_cfsav1=True, cfsav1_gate_ramp_epochs=20))
        trainer.model = nn.Sequential(CFSAv1(8))
        trainer.ema = ModelEMA(trainer.model)
        trainer.epoch = 0
        trainer._update_cfsav1_gate_scale()
        # Exactly the dtype conversion performed by the AMP training validator.
        with torch.inference_mode():
            trainer.ema.ema.half().float()
        self.assertTrue(trainer.ema.ema[0].gate_scale.is_inference())
        trainer.epoch = 1
        trainer._update_cfsav1_gate_scale()
        self.assertFalse(trainer.ema.ema[0].gate_scale.is_inference())
        self.assertIn('gate_scale', dict(trainer.ema.ema[0].named_buffers()))
        self.assertAlmostEqual(trainer.ema.ema[0].gate_scale.item(), .145, places=6)
        trainer.model(torch.randn(2,8,7,9)).square().mean().backward()
        trainer.ema.update(trainer.model)
        trainer._update_cfsav1_gate_scale(ema_only=True)
        torch.testing.assert_close(trainer.ema.ema[0].gate_scale,trainer.model[0].gate_scale,rtol=0,atol=0)

    def test_actual_training_validation_checkpoint_and_resume(self):
        self._check_training_validation_checkpoint_and_resume(device='cpu', amp=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_training_validation_next_epoch_and_resume(self):
        # Skip only the unrelated detector-based AMP preflight (may download data).
        # Actual autocast, GradScaler, half-precision validation and EMA are exercised.
        from unittest.mock import patch
        with patch('vajra.core.trainer.check_amp', return_value=True):
            self._check_training_validation_checkpoint_and_resume(device='0', amp=True)

    def _check_training_validation_checkpoint_and_resume(self, device, amp):
        import cv2
        import numpy as np
        import yaml

        with tempfile.TemporaryDirectory(prefix='cfsav1-ramp-') as directory:
            root = Path(directory)
            for split, count in [('train',4),('val',2)]:
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                for i in range(count):
                    image = np.full((64,64,3),30+i*10,dtype=np.uint8)
                    image[16:48,16:48] = 180
                    cv2.imwrite(str(root/'images'/split/f'{i}.png'),image)
                    (root/'labels'/split/f'{i}.txt').write_text('0 .25 .25 .75 .25 .75 .75 .25 .75\n')
            data=root/'data.yaml'
            data.write_text(yaml.safe_dump(dict(path=root.as_posix(),train='images/train',val='images/val',names={0:'defect'})))
            settings=dict(model='vajra-v1-nano-seg',data=str(data),device=device,amp=amp,
                          img_size=64,batch=2,workers=0,epochs=4,optimizer='AdamW',save_period=1,
                          project=str(root/'runs'),name='ramp',pretrained=False,plots=False,verbose=False,
                          mosaic=0.,close_mosaic=0,use_cfsav1=True,cfsav1_gate_ramp_epochs=4,
                          cfsav1_gate_ramp_start=.1,cfsav1_warmup_epochs=0)
            trainer=SegmentationTrainer(model_configuration=settings)
            initial=[]
            seen=[]

            def snapshot(t):
                if t.epoch==0:
                    initial.append(next(t.model.model[0].parameters()).detach().clone())

            def check_epoch(t):
                self.assertEqual(t.amp, amp)
                expected=.1+.9*min(t.epoch/4,1)
                for model in [t.model,t.ema.ema]:
                    self.assertAlmostEqual(model.model[-1].proto_enhance.gate_scale.item(),expected,places=6)
                self.assertFalse(getattr(t,'_cfsav1_warmup_active',False))
                seen.append((t.epoch,expected))
                if t.epoch==0:
                    self.assertTrue(next(t.model.model[0].parameters()).requires_grad)
                    # AMP can legitimately skip these two initial batches while
                    # backing off its loss scale. CPU verifies actual updates;
                    # CUDA verifies the real validation -> next-epoch path.
                    if not amp:
                        self.assertFalse(torch.equal(initial[0],next(t.model.model[0].parameters())))
                if t.epoch==1:
                    t.stop=True

            trainer.callbacks['on_train_epoch_start'].append(snapshot)
            trainer.callbacks['on_train_epoch_end'].append(check_epoch)
            trainer.train()
            self.assertEqual([e for e,_ in seen],[0,1])
            periodic=next(Path(trainer.weights_dir).glob('epoch1-*.pt'))
            checkpoint=torch.load(periodic,map_location='cpu',weights_only=False)
            self.assertAlmostEqual(checkpoint['ema'].model[-1].proto_enhance.gate_scale.item(),.325,places=3)
            resumed=SegmentationTrainer(model_configuration=dict(model=str(periodic),resume=str(periodic),device=device,workers=0))
            resumed.callbacks['on_train_epoch_end'].append(check_epoch)
            resumed.train()
            self.assertEqual(resumed.start_epoch,2)
            self.assertEqual([e for e,_ in seen],[0,1,2,3])
            self.assertEqual(resumed.args.cfsav1_gate_ramp_epochs,4)
            # Saving/standalone final validation must keep an early checkpoint's scale.
            last=torch.load(resumed.last,map_location='cpu',weights_only=False)
            self.assertAlmostEqual(last['model'].model[-1].proto_enhance.gate_scale.item(),.775,places=3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_with_nonunit_scale(self):
        module=CFSAv1(64).cuda()
        module.gate_scale.fill_(.1)
        x=torch.randn(2,64,15,17,device='cuda',requires_grad=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=module(x).float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(),0)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(module.half()(x.detach().half())).all())


if __name__=='__main__':
    unittest.main()
