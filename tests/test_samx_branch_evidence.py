import io
import unittest

import torch
from torch import nn

from samx_branch_experiment_config import decide_g2, options
from vajra.configs import get_config
from vajra.core.model import Model
from vajra.core.trainer import Trainer
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.modules import LDConv
from vajra.nn.samx import LocalGlobalSAMX
from vajra.nn.samx_branch_evidence import BranchEvidenceSAMX


class BranchEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def build(mode):
        args = get_config(model_configuration=options(mode))
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = args, {"nc": 4}
        torch.manual_seed(42)
        model = trainer.get_model("vajra-v1-nano-seg", verbose=False)
        model.args = args
        return model, args

    @staticmethod
    def sample(device="cpu"):
        masks = torch.zeros(2, 16, 16, device=device)
        masks[:, 4:12, 4:12] = 1
        return dict(img=torch.rand(2, 3, 64, 64, device=device), masks=masks,
                    batch_idx=torch.tensor([0., 1.], device=device), cls=torch.zeros(2, 1, device=device),
                    bboxes=torch.tensor([[.5, .5, .5, .5]] * 2, device=device))

    def test_exact_localglobal_recovery_and_rng(self):
        torch.manual_seed(42)
        baseline = LocalGlobalSAMX(16, 16, 16, 8, rep_csp=True).eval()
        next_random = torch.rand(8)
        torch.manual_seed(42)
        candidate = BranchEvidenceSAMX(16, 16, 16, 8, rep_csp=True).eval()
        torch.testing.assert_close(torch.rand(8), next_random, atol=0, rtol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, atol=0, rtol=0)
        for h, w in ((1, 1), (1, 7), (5, 1), (7, 11)):
            x = torch.randn(2, 16, h, w)
            with torch.inference_mode():
                torch.testing.assert_close(candidate(x), baseline(x), atol=0, rtol=0)
                torch.testing.assert_close(candidate.forward_split(x), baseline(x), atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in candidate.evidence_route.parameters()), 458)

    def test_statistics_use_both_branches_and_disagreement(self):
        r = torch.tensor([[[[1., -2.]], [[3., 4.]]]])
        d = torch.tensor([[[[2., -1.]], [[-1., 2.]]]])
        expected = torch.cat((r.mean(1, True), r.amax(1, True), d.mean(1, True), d.amax(1, True),
                              (r-d).abs().mean(1, True), (r-d).abs().amax(1, True)), 1)
        torch.testing.assert_close(BranchEvidenceSAMX.branch_statistics(r, d), expected, atol=0, rtol=0)

    def test_shared_head_reads_actual_stage_outputs_once_per_path(self):
        module = BranchEvidenceSAMX(16, 16, 16, 8, rep_csp=True).eval()
        records, calls, handles = {}, [], []
        for stage in (2, 3):
            block = getattr(module, f"block{stage}")
            for path in ("regular_path", "deform_path"):
                def save(m, inputs, output, label=(stage, path)):
                    self.assertNotIn(label, records)
                    records[label] = output.detach()
                handles.append(getattr(block, path).register_forward_hook(save))
        handles.append(module.evidence_route.register_forward_pre_hook(lambda m, inputs: calls.append(inputs[0].detach())))
        module(torch.randn(2, 16, 7, 11))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(calls), 2)
        for index, stage in enumerate((2, 3)):
            torch.testing.assert_close(calls[index], module.branch_statistics(records[stage, "regular_path"],
                                                                               records[stage, "deform_path"]), atol=0, rtol=0)
        self.assertFalse(torch.equal(calls[0], calls[1]))

    def test_zero_init_has_live_gradient_and_upstream_head_learns_after_step(self):
        torch.manual_seed(4)
        module = BranchEvidenceSAMX(16, 16, 16, 8, rep_csp=True).train()
        optimizer = torch.optim.SGD(module.parameters(), lr=.01)
        x = torch.randn(2, 16, 7, 11)
        module(x).square().mean().backward()
        self.assertGreater(module.evidence_route[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.evidence_route[0].weight.grad.count_nonzero().item(), 0)
        optimizer.step(); optimizer.zero_grad()
        module(x).square().mean().backward()
        self.assertGreater(module.evidence_route[0].weight.grad.abs().sum().item(), 0)
        for parameter in module.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_learned_correction_changes_weights_and_zero_hook_recovers_old_fusion(self):
        module = BranchEvidenceSAMX(16, 16, 16, 8, rep_csp=True).eval()
        x = torch.randn(2, 16, 7, 11)
        with torch.no_grad():
            reference = module(x)
            module.evidence_route[-1].weight.normal_(0, .2)
            module.evidence_route[-1].bias.copy_(torch.tensor([.2, -.2]))
            changed = module(x)
        self.assertGreater((reference-changed).abs().max().item(), 0)
        handle = module.evidence_route.register_forward_hook(lambda m, inputs, output: torch.zeros_like(output))
        with torch.no_grad():
            torch.testing.assert_close(module(x), reference, atol=0, rtol=0)
        handle.remove()
        with torch.no_grad():
            torch.testing.assert_close(module(x), changed, atol=0, rtol=0)

    def test_full_model_common_initialization_and_rng(self):
        g1, _ = self.build("g1_p4_only")
        g1.eval()
        next_random = torch.rand(8)
        args = get_config(model_configuration={**options("g1_p4_only"), "samx_mode": "local_global"})
        trainer = SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args, trainer.data = args, {"nc": 4}
        torch.manual_seed(42)
        baseline = trainer.get_model("vajra-v1-nano-seg", verbose=False).eval()
        torch.testing.assert_close(torch.rand(8), next_random, atol=0, rtol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(g1.state_dict()[name], value, atol=0, rtol=0)
        x = torch.rand(2, 3, 96, 96)
        with torch.inference_mode():
            torch.testing.assert_close(g1(x), baseline(x), atol=0, rtol=0)
        g2, _ = self.build("g2_p4_proto_free")
        g2.eval()
        for name, value in g1.state_dict().items():
            torch.testing.assert_close(g2.state_dict()[name], value, atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(g2(x), g1(x), atol=0, rtol=0)

    def test_real_loss_optimizer_pickle_and_checkpoint_options(self):
        for mode in ("g1_p4_only", "g2_p4_proto_free"):
            model, args = self.build(mode)
            batch = self.sample()
            loss, items = model(batch)
            self.assertTrue(torch.isfinite(loss).all()); loss.backward()
            self.assertFalse(model.criterion.use_morph_mask_loss)
            neck = model.model[0].vajra_neck3
            self.assertGreater(neck.evidence_route[-1].weight.grad.abs().sum().item(), 0)
            trainer = Trainer.__new__(Trainer); trainer.args = args
            optimizer = trainer.build_optimizer(model, name="SGD", lr=.01, decay=.0005)
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            offsets = {id(p) for m in model.modules() if isinstance(m, LDConv) for p in m.p_conv.parameters()}
            self.assertEqual(offsets, {id(p) for group in optimizer.param_groups if group.get("lr_scale") == .1 for p in group["params"]})
            self.assertTrue({id(p) for p in neck.evidence_route.parameters()}.isdisjoint(offsets))
            for m in model.modules():
                if isinstance(m, LDConv):
                    self.assertTrue(torch.isfinite(m.p_conv.weight.grad).all())
                    self.assertGreater(m.p_conv.weight.grad.abs().sum().item(), 0)
            stream = io.BytesIO(); torch.save(model.eval(), stream); stream.seek(0)
            reloaded = torch.load(stream, weights_only=False).eval()
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(batch["img"]), model(batch["img"]), atol=0, rtol=0)
            self.assertEqual(Model._reset_checkpoint_args(vars(args))["samx_mode"], "branch_evidence")

    def test_g2_decision_is_predeclared_and_checks_all_metrics(self):
        baseline = dict(metrics={"metrics/mAP50-95(Mask)": .26, "metrics/mAP50(Mask)": .61,
                                 "metrics/recall(Mask)": .60}, boundary={"overall": {"f1": .50}})
        import copy
        self.assertFalse(decide_g2(baseline, baseline)["run_g2"])
        candidate = copy.deepcopy(baseline)
        candidate["metrics"]["metrics/mAP50-95(Mask)"] += .003
        self.assertTrue(decide_g2(baseline, candidate)["run_g2"])
        for key in ("metrics/mAP50(Mask)", "metrics/recall(Mask)"):
            changed = copy.deepcopy(candidate); changed["metrics"][key] -= .00101
            self.assertFalse(decide_g2(baseline, changed)["run_g2"])
        changed = copy.deepcopy(candidate); changed["boundary"]["overall"]["f1"] -= .00101
        self.assertFalse(decide_g2(baseline, changed)["run_g2"])
        for mode in ("g1_p4_only", "g2_p4_proto_free"):
            args = options(mode)
            self.assertFalse(args["use_cfsa"] or args["use_cfsav1"] or args["samx_proto_share_coeff"])
            self.assertEqual(args["cfsav1_gate_ramp_epochs"], 0)
            self.assertFalse(args["pretrained"] or args["resume"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_backward_half_inference_and_fusion(self):
        for mode in ("g1_p4_only", "g2_p4_proto_free"):
            model, _ = self.build(mode)
            model.cuda().train(); batch = self.sample("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            gradient = model.model[0].vajra_neck3.evidence_route[-1].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            model.float().eval()
            with torch.inference_mode():
                expected = model(batch["img"])
                self.assertTrue(torch.isfinite(expected[0]).all())
                model.fuse(verbose=False)
                torch.testing.assert_close(model(batch["img"]), expected, atol=2e-5, rtol=2e-4)
                model.half()
                self.assertTrue(torch.isfinite(model(batch["img"].half())[0]).all())


if __name__ == "__main__":
    unittest.main()
