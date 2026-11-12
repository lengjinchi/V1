"""Full validation, branch-gate interventions, and interleaved forward timing."""
import gc
import hashlib
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from diagnose_cfsav1_regions import valid_mask
from samx_followup_validation import BoundarySegmentationValidator
from vajra.metrics import box_iou, mask_iou
from vajra.nn.samx_branch_evidence import BranchEvidenceSAMX


def greedy_pairs(matrix, gt_classes, pred_classes):
    candidates = ((matrix >= .5) & (gt_classes[:, None] == pred_classes[None])).nonzero().cpu().tolist()
    values = matrix.detach().cpu()
    candidates.sort(key=lambda ij: (-float(values[ij[0], ij[1]]), *ij))
    used_gt, used_pred, pairs = set(), set(), []
    for g, p in candidates:
        if g not in used_gt and p not in used_pred:
            pairs.append((g, p)); used_gt.add(g); used_pred.add(p)
    return pairs


class GateObserver:
    def __init__(self, model, zero=False):
        modules = [m for m in model.modules() if isinstance(m, BranchEvidenceSAMX)]
        assert len(modules) == 1
        module = modules[0]
        self.zero, self.batch, self.base, self.global_logits = zero, None, None, None
        self.stage, self.statistics = 0, {}
        self.handles = [module.route.register_forward_hook(self.read_global),
                        module.local_route.register_forward_hook(self.read_local),
                        module.evidence_route.register_forward_hook(self.read_correction)]

    def read_global(self, module, inputs, output):
        self.global_logits = output.detach()

    def read_local(self, module, inputs, output):
        self.base = self.global_logits + output.detach()
        self.stage = 0

    def read_correction(self, module, inputs, output):
        correction = torch.zeros_like(output) if self.zero else output
        alpha = (self.base + correction.detach()).float().softmax(1)[:, 1:2]
        assert self.batch is not None and self.stage < 2
        name = f"stage{self.stage+2}"
        self.stage += 1
        valid = valid_mask(self.batch, alpha.shape[-2:])
        # Max projection retains any labelled foreground in a P4 cell.
        foreground = F.adaptive_max_pool2d((self.batch["masks"] > 0).float()[:, None], alpha.shape[-2:]) > 0
        foreground &= valid
        background = valid & ~foreground
        count = valid.sum((-2, -1)).clamp_min(1)
        mean = (alpha * valid).sum((-2, -1)) / count
        spatial_std = (((alpha-mean[:, :, None, None]).square()*valid).sum((-2, -1))/count).sqrt()
        values = torch.stack([valid.sum(), (alpha*valid).sum(), spatial_std.sum(),
                              (correction.detach().float().square()*valid).sum(),
                              foreground.sum(), (alpha*foreground).sum(), background.sum(), (alpha*background).sum()]).cpu().tolist()
        item = self.statistics.setdefault(name, dict(images=0, cells=0, alpha_sum=0., spatial_std_sum=0.,
            correction_squared_sum=0., foreground_cells=0, foreground_alpha_sum=0., background_cells=0, background_alpha_sum=0.))
        item["images"] += alpha.shape[0]
        for key, value in zip(("cells", "alpha_sum", "spatial_std_sum", "correction_squared_sum", "foreground_cells",
                               "foreground_alpha_sum", "background_cells", "background_alpha_sum"), values):
            item[key] += value
        return correction if self.zero else None

    def report(self):
        return {name: {**s, "ld_weight_mean": s["alpha_sum"]/max(s["cells"], 1),
                       "mean_spatial_std": s["spatial_std_sum"]/max(s["images"], 1),
                       "correction_logit_rms": (s["correction_squared_sum"]/max(2*s["cells"], 1))**.5,
                       "foreground_ld_weight": s["foreground_alpha_sum"]/max(s["foreground_cells"], 1),
                       "background_ld_weight": s["background_alpha_sum"]/max(s["background_cells"], 1)}
                for name, s in self.statistics.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.batch = self.base = self.global_logits = None


class EvidenceValidator(BoundarySegmentationValidator):
    def __init__(self, *args, zero=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero, self.observer = zero, None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.observer = GateObserver(model, self.zero)
        self.error_counts = dict(gt=0, predictions=0, box_matched=0, box_missed=0,
                                 box_matched_mask_failed_05=0, box_matched_mask_passed_075=0,
                                 independent_mask_matched=0)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self.observer.batch = batch
        return batch

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            prepared = self._prepare_batch(si, batch)
            classes = prepared["cls"].long()
            gt = prepared["masks"] == (torch.arange(len(classes), device=self.device)[:, None, None]+1)
            selected = pred[pred[:, 4] >= .25]
            if len(selected):
                predn, pm = self._prepare_pred(selected, prepared, proto)
            else:
                predn, pm = selected, torch.zeros(0, *gt.shape[-2:], device=self.device)
            if len(gt) and gt.shape[-2:] != pm.shape[-2:]:
                gt = F.interpolate(gt[None].float(), pm.shape[-2:], mode="bilinear", align_corners=False)[0] > .5
            bm = box_iou(prepared["bbox"], predn[:, :4])
            mm = (mask_iou(gt.flatten(1).float(), pm.flatten(1).float()) if len(gt) and len(pm)
                  else torch.zeros(len(gt), len(pm), device=self.device))
            bp, mp = greedy_pairs(bm, classes, predn[:, 5]), greedy_pairs(mm, classes, predn[:, 5])
            values = mm.detach().cpu()
            e = self.error_counts
            e["gt"] += len(gt); e["predictions"] += len(predn)
            e["box_matched"] += len(bp); e["box_missed"] += len(gt)-len(bp)
            e["box_matched_mask_failed_05"] += sum(float(values[g, p]) < .5 for g, p in bp)
            e["box_matched_mask_passed_075"] += sum(float(values[g, p]) >= .75 for g, p in bp)
            e["independent_mask_matched"] += len(mp)

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if self.observer:
                self.observer.close()


def evaluate_evidence(checkpoint, args, zero=False, progress=None):
    from vajra.utils.torch_utils import init_seeds
    init_seeds(42, deterministic=True)
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validator = EvidenceValidator(args={**args, "task": "segment", "mode": "val"}, zero=zero)
    if progress:
        validator.add_callback("on_val_batch_end", progress)
    metrics = validator(model=str(checkpoint))
    assert validator.seen == 1532 and int(validator.nt_per_class.sum()) == 3942
    assert sum(s["images"] for s in validator.observer.statistics.values()) == 3064
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
        counts=dict(images=1532, instances=3942), metrics={k: float(v) for k, v in metrics.items()},
        boundary=validator.boundary.report(), gate_statistics=validator.observer.report(),
        gate_statistics_protocol="Padding excluded; foreground is max-projected union of GT masks. Statistics are descriptive, not proof of useful routing.",
        error_counts_conf025=validator.error_counts,
        error_protocol="Post-NMS conf=0.25; same-class greedy one-to-one descending Box IoU>=0.5; conditional masks use the box-paired prediction; not official macro Recall.",
        per_class={str(int(c)): dict(name=validator.names[int(c)], instances=int(validator.nt_per_class[c]),
            **dict(zip(("precision", "recall", "map50", "map95"), map(float, validator.metrics.seg.class_aware_results(i)))))
            for i, c in enumerate(validator.metrics.seg.ap_class_idx)},
        intervention="zero_evidence_logits" if zero else "learned", validation_options=args)
    assert result["error_counts_conf025"]["gt"] == 3942
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    result["checkpoint_unchanged"] = True
    del validator
    gc.collect(); torch.cuda.empty_cache()
    return result


def benchmark_pair(baseline_checkpoint, candidate_checkpoint):
    from vajra import Vajra
    gc.collect(); torch.cuda.empty_cache()
    paths = {"baseline": Path(baseline_checkpoint), "candidate": Path(candidate_checkpoint)}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    models = {k: Vajra(str(p)).model.cuda().float().eval().fuse(verbose=False) for k, p in paths.items()}
    x = torch.zeros(1, 3, 288, 1632, device="cuda")
    times = {key: [] for key in models}
    with torch.inference_mode():
        for model in models.values():
            for _ in range(50): model(x)
        torch.cuda.synchronize()
        for order in (("baseline", "candidate"), ("candidate", "baseline"), ("baseline", "candidate")):
            for key in order:
                for _ in range(10): models[key](x)
                for _ in range(100):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); models[key](x); end.record(); end.synchronize()
                    times[key].append(start.elapsed_time(end))
    result = dict(precision="FP32", shape=[1, 3, 288, 1632], warmup=50, rounds=3, iterations_per_round=100,
        measurements={k: dict(mean_ms=statistics.mean(v), fps=1000/statistics.mean(v)) for k, v in times.items()},
        note="Same-process interleaved pure forward; excludes data, NMS and mask reconstruction.")
    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[key]
    del models, x
    gc.collect(); torch.cuda.empty_cache()
    return result
