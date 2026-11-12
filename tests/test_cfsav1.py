"""Focused CFSAv1 regression tests; run with unittest, no dataset/downloads."""

import copy
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vajra.configs import get_config
from vajra.core.model import Model as PublicModel
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.cfsav1 import CFSAv1, LocalBandDecomposition
from vajra.nn.head import Segmentation
from vajra.nn.modules import FFTGuidedEMA
from vajra.nn.vajra import SegmentationModel


def options(**kwargs):
    return get_config(model_configuration={"task": "segment", **kwargs})


def batch(overlap=True, empty=False):
    n, size = 2, 64
    masks = torch.zeros(n if overlap or not empty else 0, size // 4, size // 4)
    if not empty:
        masks[:, 4:12, 4:12] = 1
    return {"img": torch.rand(n, 3, size, size), "masks": masks,
            "batch_idx": torch.empty(0) if empty else torch.tensor([0., 1.]),
            "cls": torch.empty(0, 1) if empty else torch.zeros(n, 1),
            "bboxes": torch.empty(0, 4) if empty else torch.tensor([[.5, .5, .5, .5]] * n),
            "im_file": ["a.png", "b.png"]}


class CFSAv1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reconstruction_identity_and_small_odd_shapes(self):
        for size in ((1, 1), (3, 7), (17, 19)):
            x = torch.randn(2, 8, *size)
            parts = LocalBandDecomposition()(x)
            torch.testing.assert_close(parts.sum(2), x)
            for global_context in (False, True):
                model = CFSAv1(8, groups=4, global_context=global_context)
                self.assertTrue(torch.equal(model(x), x))
                model(x).square().mean().backward()
                self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)


    def test_smoothed_gate_identity_gradient_and_legacy(self):
        for size in ((1, 1), (3, 7), (17, 19)):
            model = CFSAv1(8, rho=.1, gate_smooth=True)
            x = torch.randn(2, 8, *size, requires_grad=True)
            self.assertTrue(torch.equal(model(x), x))
            model(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)
        original = CFSAv1(8)
        nn.init.normal_(original.gate[-1].weight, std=.1)
        legacy = copy.deepcopy(original)
        del legacy.gate_smooth
        x = torch.randn(2, 8, 5, 7)
        torch.testing.assert_close(legacy(x), original(x), rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        self.assertTrue(restored.gate_smooth)
        torch.testing.assert_close(restored(x), model(x))

    def test_smoothed_gate_matches_local_coefficient_average(self):
        class FixedGate(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = nn.Parameter(logits)

            def forward(self, condition):
                return self.logits.expand(condition.shape[0], -1, -1, -1)

        # Spatially alternating gates expose placement before/after tanh, and
        # explicit window means check that padding does not dilute edge gains.
        x = torch.randn(2, 8, 5, 7)
        logits = (torch.arange(35).reshape(1, 1, 5, 7) % 3 - 1).float() * 2
        model = CFSAv1(8, rho=.1, gate_smooth=True)
        model.gate = FixedGate(logits.repeat(1, 12, 1, 1))
        raw = .1 * model.gate.logits.tanh()
        expected_delta = torch.empty_like(raw)
        for y in range(5):
            for z in range(7):
                expected_delta[..., y, z] = raw[..., max(0,y-1):y+2, max(0,z-1):z+2].mean((-2,-1))
        bands = model.bands(x)
        gains = expected_delta.reshape(1, 4, 1, 3, 5, 7).expand(2, 4, 2, 3, 5, 7).reshape(2, 8, 3, 5, 7)
        expected = x + (gains * bands).sum(2)
        torch.testing.assert_close(model(x), expected)
        self.assertLessEqual(expected_delta.abs().max().item(), .1)
        model(x).square().mean().backward()
        self.assertGreater(model.gate.logits.grad.abs().sum().item(), 0)
        model.gate_smooth = False
        self.assertFalse(torch.allclose(model(x), expected))



    def test_switches_and_legacy_checkpoint_head(self):
        head = Segmentation(2, in_channels=[64, 128, 256])
        self.assertIsInstance(head.proto_enhance, nn.Identity)
        head.configure_enhancement(use_cfsa=True)
        self.assertIsInstance(head.proto_enhance, FFTGuidedEMA)
        legacy = Segmentation(2, in_channels=[64, 128, 256], use_fft_guided_ema_proto_only=True)
        legacy.proto_enhance.load_state_dict(head.proto_enhance.state_dict())
        x = torch.randn(2, 64, 8, 8)
        torch.testing.assert_close(legacy.proto_enhance(x), head.proto_enhance(x))
        del legacy.use_cfsav1  # old pickled heads have no new flag
        self.assertIn("CFSAv1=False", legacy.get_module_info()[1])
        head.configure_enhancement(use_cfsav1=True)
        self.assertIsInstance(head.proto_enhance, CFSAv1)
        head.configure_enhancement()
        self.assertIsInstance(head.proto_enhance, nn.Identity)
        with self.assertRaises(ValueError):
            head.configure_enhancement(use_cfsav1=True, use_cfsa=True)
        for invalid in ({"use_cfsa": True, "use_cfsav1": True},
                        {"cfsav1_sensitivity": True}, {"cfsav1_rho": 1}, {"cfsav1_groups": 0}):
            with self.assertRaises(ValueError):
                options(**invalid)

    def test_student_loss_backward_and_inference_contract(self):
        model = SegmentationModel(num_classes=2, verbose=False, use_cfsav1=True, cfsav1_global_context=True)
        model.args = options(use_cfsav1=True, cfsav1_global_context=True)
        model.train()
        sample = batch()
        total, items = model(sample)
        self.assertEqual(items.shape, (4,))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        module = model.model[-1].proto_enhance
        self.assertFalse(hasattr(module, "sensitivity_head"))
        self.assertGreater(module.global_gate[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            result = model(sample["img"])
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[1]), 3)
            validation_loss, validation_items = model.loss(batch(), result)
            self.assertTrue(torch.isfinite(validation_loss))
            self.assertEqual(validation_items.shape, (4,))
        self.assertFalse(any("reference" in k for k in model.state_dict()))
        checkpoint = io.BytesIO()
        torch.save(model, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=False).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(sample["img"])[0], result[0])


    def test_public_checkpoint_options_and_trainer_weight_transfer(self):
        args = {"use_cfsav1": True, "cfsav1_groups": 4, "cfsav1_sensitivity": False,
                "cfsav1_gate_smooth": True, "lr0": .1}
        preserved = PublicModel._reset_checkpoint_args(args)
        self.assertTrue(preserved["use_cfsav1"])
        self.assertTrue(preserved["cfsav1_gate_smooth"])
        self.assertNotIn("lr0", preserved)
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args = options(use_cfsav1=True, cfsav1_gate_smooth=True)
        trainer.data = {"nc": 2}
        model = trainer.get_model("vajra-v1-nano-seg", verbose=False)
        self.assertTrue(model.model[-1].proto_enhance.gate_smooth)
        with torch.no_grad():
            model.model[-1].proto_enhance.gate[-1].bias.fill_(.3)
        restored = trainer.get_model("some-checkpoint.pt", weights=model, verbose=False)
        self.assertTrue(restored.model[-1].proto_enhance.gate_smooth)
        torch.testing.assert_close(restored.model[-1].proto_enhance.gate[-1].bias,
                                   model.model[-1].proto_enhance.gate[-1].bias)

    def test_full_trainer_on_synthetic_dataset(self):
        import cv2
        import numpy as np
        import yaml

        with tempfile.TemporaryDirectory(prefix="cfsav1-train-") as directory:
            root = Path(directory)
            for split, count in (("train", 4), ("val", 2)):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                for i in range(count):
                    image = np.full((64, 64, 3), 40 + i * 10, dtype=np.uint8)
                    image[16:48, 16:48] = (120, 160, 180)
                    cv2.imwrite(str(root / "images" / split / f"{i}.png"), image)
                    (root / "labels" / split / f"{i}.txt").write_text(
                        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                           "val": "images/val", "names": {0: "defect"}}), encoding="utf-8")
            observed = []

            def warmup_start(trainer):
                if trainer.epoch == 0:
                    observed.append(copy.deepcopy(trainer.model.model[0].state_dict()))

            def warmup_end(trainer):
                if trainer.epoch == 0:
                    current = trainer.model.model[0].state_dict()
                    for name, tensor in observed[0].items():
                        torch.testing.assert_close(current[name], tensor, rtol=0, atol=0)
                if trainer.epoch == 1:
                    trainer.stop = True  # leave an unstripped periodic checkpoint to resume

            trainer = SegmentationTrainer(model_configuration={
                "model": "vajra-v1-nano-seg", "data": str(data), "device": "cpu", "amp": False,
                "img_size": 64, "batch": 2, "workers": 0, "epochs": 3, "optimizer": "AdamW", "save_period": 1,
                "project": str(root / "runs"), "name": "integration", "pretrained": False,
                "plots": False, "verbose": False, "mosaic": 0., "close_mosaic": 0,
                "use_cfsav1": True,
                "cfsav1_gate_smooth": True,
                "cfsav1_global_context": True,
                "cfsav1_warmup_epochs": 1,
            })
            trainer.callbacks["on_train_epoch_start"].append(warmup_start)
            trainer.callbacks["on_train_epoch_end"].append(warmup_end)
            trainer.train()
            self.assertEqual(len(observed), 1)
            self.assertTrue(trainer.last.is_file())
            self.assertFalse((Path(trainer.save_dir) / "cfsav1_reference_stats.pt").exists())
            self.assertNotIn("sensitivity_loss", Path(trainer.csv).read_text())
            checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
            student = checkpoint["ema"] or checkpoint["model"]
            self.assertIsInstance(student.model[-1].proto_enhance, CFSAv1)
            self.assertTrue(student.model[-1].proto_enhance.gate_smooth)
            self.assertTrue(student.model[-1].proto_enhance.global_context)
            self.assertFalse(any("reference" in key for key in student.state_dict()))
            trainer.plot_metrics()
            self.assertTrue((Path(trainer.save_dir) / "results.png").is_file())
            self.assertFalse(hasattr(trainer, "cfsav1_reference"))
            periodic = next(Path(trainer.weights_dir).glob("epoch1-*.pt"))
            resumed = SegmentationTrainer(model_configuration={
                "model": str(periodic), "resume": str(periodic), "device": "cpu", "workers": 0,
            })
            resumed.train()
            self.assertEqual(resumed.start_epoch, 2)
            self.assertEqual(resumed.epoch, 2)
            self.assertTrue(resumed.model.model[-1].proto_enhance.gate_smooth)
            self.assertTrue(resumed.model.model[-1].proto_enhance.global_context)

    def test_traced_module_handles_another_spatial_shape(self):
        module = CFSAv1(8, global_context=True, gate_smooth=True).eval()
        nn.init.normal_(module.gate[-1].weight, std=.03)
        traced = torch.jit.trace(module, torch.randn(1, 8, 9, 11))
        sample = torch.randn(2, 8, 13, 15)
        torch.testing.assert_close(traced(sample), module(sample))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_and_half_inference(self):
        model = CFSAv1(64, global_context=True, gate_smooth=True).cuda()
        x = torch.randn(2, 64, 15, 17, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(x)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model.half()(x.detach().half())).all())


if __name__ == "__main__":
    unittest.main()
