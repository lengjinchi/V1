"""Additional boundary evaluation and matched forward-speed measurements."""
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from vajra.metrics import mask_iou
from vajra.models.vajra.segment.val import SegmentationValidator


def mask_boundaries(masks):
    if not len(masks):
        return masks.bool()
    masks=masks.bool()
    eroded=1-F.max_pool2d(F.pad(1-masks[:,None].float(),(1,1,1,1),value=1),3,1)
    return masks & ~(eroded[:,0]>.5)


class BoundaryAccumulator:
    """Class-aware Mask-IoU matches, counting unmatched boundary pixels too."""
    def __init__(self,confidence=.25,iou=.5,tolerance=1):
        self.confidence,self.iou,self.tolerance=confidence,iou,tolerance
        self.classes={}

    def update(self,pred_masks,pred_classes,scores,gt_masks,gt_classes):
        selected=scores>=self.confidence
        pred_masks,pred_classes=pred_masks[selected].bool(),pred_classes[selected].long()
        gt_masks,gt_classes=gt_masks.bool(),gt_classes.long()
        pb,gb=mask_boundaries(pred_masks),mask_boundaries(gt_masks)
        pairs=[]
        if len(pred_masks) and len(gt_masks):
            matrix=mask_iou(gt_masks.flatten(1).float(),pred_masks.flatten(1).float())
            candidates=(matrix>=self.iou)&(gt_classes[:,None]==pred_classes[None])
            indices=candidates.nonzero().cpu().tolist()
            values=matrix.detach().cpu()
            indices.sort(key=lambda p:(-float(values[p[0],p[1]]),p[0],p[1]))
            used_gt,used_pred=set(),set()
            for g,p in indices:
                if g not in used_gt and p not in used_pred:
                    pairs.append((g,p));used_gt.add(g);used_pred.add(p)
        k=2*self.tolerance+1
        pd=(F.max_pool2d(pb[:,None].float(),k,1,self.tolerance)[:,0]>0) if len(pb) else pb
        gd=(F.max_pool2d(gb[:,None].float(),k,1,self.tolerance)[:,0]>0) if len(gb) else gb
        for c in torch.cat((pred_classes,gt_classes)).unique().cpu().tolist():
            pindex,gindex=pred_classes==c,gt_classes==c
            counts=[pb[pindex].sum(),gb[gindex].sum(),pb.new_zeros((),dtype=torch.long),gb.new_zeros((),dtype=torch.long)]
            matched=0
            for g,p in pairs:
                if int(gt_classes[g])==c:
                    counts[2]=counts[2]+(pb[p]&gd[g]).sum()
                    counts[3]=counts[3]+(gb[g]&pd[p]).sum()
                    matched+=1
            values=torch.stack(counts+[pindex.sum(),gindex.sum()]).cpu().tolist()
            acc=self.classes.setdefault(str(c),dict(pred_pixels=0,gt_pixels=0,
                correct_pred_pixels=0,correct_gt_pixels=0,pred_instances=0,gt_instances=0,matched_instances=0))
            for key,value in zip(('pred_pixels','gt_pixels','correct_pred_pixels','correct_gt_pixels',
                                  'pred_instances','gt_instances'),values):
                acc[key]+=int(value)
            acc['matched_instances']+=matched

    def report(self):
        keys=('pred_pixels','gt_pixels','correct_pred_pixels','correct_gt_pixels',
              'pred_instances','gt_instances','matched_instances')
        def finish(counts):
            p=counts['correct_pred_pixels']/counts['pred_pixels'] if counts['pred_pixels'] else 0.
            r=counts['correct_gt_pixels']/counts['gt_pixels'] if counts['gt_pixels'] else 0.
            return dict(**counts,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)
        return dict(overall=finish({k:sum(c[k] for c in self.classes.values()) for k in keys}),
            per_class={k:finish(v) for k,v in self.classes.items()},
            protocol=dict(confidence=self.confidence,mask_iou_matching=self.iou,
                matching='class-aware greedy descending mask IoU, one-to-one',
                tolerance_mask_pixels=self.tolerance,neighborhood='square/Chebyshev',
                mask_ratio=4,nominal_tolerance_input_pixels=4*self.tolerance,
                unmatched_predictions_and_gt_included=True))


class BoundarySegmentationValidator(SegmentationValidator):
    def init_metrics(self,model):
        super().init_metrics(model)
        self.boundary=BoundaryAccumulator()

    def update_metrics(self,preds,batch):
        super().update_metrics(preds,batch)
        for si,(pred,proto) in enumerate(zip(preds[0],preds[1])):
            prepared=self._prepare_batch(si,batch)
            classes=prepared['cls']
            if self.args.overlap_mask:
                index=torch.arange(len(classes),device=self.device)[:,None,None]+1
                gt=prepared['masks']==index
            else:
                gt=prepared['masks'].bool()
            selected=pred[pred[:,4]>=.25]
            if len(selected):
                _,pm=self._prepare_pred(selected,prepared,proto)
            else:
                pm=torch.zeros(0,*gt.shape[-2:],device=gt.device,dtype=torch.bool)
            if len(gt) and gt.shape[-2:]!=pm.shape[-2:]:
                gt=F.interpolate(gt[None].float(),size=pm.shape[-2:],mode='bilinear',align_corners=False)[0]>.5
            self.boundary.update(pm,selected[:,5],selected[:,4],gt,classes)


def benchmark_forward(checkpoint):
    """FP32 batch1, rectangular input; no dataloading, NMS, or metric hooks."""
    from vajra import Vajra
    gc.collect();torch.cuda.empty_cache()
    before=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wrapper=Vajra(str(checkpoint))
    model=wrapper.model.to('cuda').float().eval().fuse(verbose=False)
    x=torch.zeros(1,3,288,1632,device='cuda')
    times=[]
    with torch.inference_mode():
        for _ in range(20):model(x)
        torch.cuda.synchronize()
        for _ in range(50):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();end.synchronize()
            times.append(start.elapsed_time(end))
    mean=statistics.mean(times)
    result=dict(precision='FP32',shape=[1,3,288,1632],warmup_iterations=20,timed_iterations=50,
                mean_ms=mean,median_ms=statistics.median(times),fps=1000/mean,
                peak_increment_gib=(torch.cuda.max_memory_allocated()-before)/2**30,
                parameters_fused=sum(p.numel() for p in model.parameters()),
                note='Pure network forward; excludes data/NMS/mask reconstruction. Not comparable to differently configured paper FPS.')
    del wrapper,model,x
    gc.collect();torch.cuda.empty_cache()
    return result


def validate_checkpoint(checkpoint,validation_options,expected=None,progress=None):
    from vajra.utils.torch_utils import init_seeds
    init_seeds(42,deterministic=True)
    checkpoint=Path(checkpoint)
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validator=BoundarySegmentationValidator(args={**validation_options,'task':'segment','mode':'val'})
    if progress is not None:
        validator.add_callback('on_val_batch_end',progress)
    metrics=validator(model=str(checkpoint))
    counts=dict(images=int(validator.seen),instances=int(validator.nt_per_class.sum()))
    assert counts==dict(images=1532,instances=3942),counts
    if expected is not None:
        for k,v in expected.items():
            assert abs(float(metrics[k])-v)<1e-7,(k,metrics[k],v)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=before,counts=counts,
        metrics={k:float(v) for k,v in metrics.items()},boundary=validator.boundary.report(),
        per_class={str(int(c)):dict(name=validator.names[int(c)],instances=int(validator.nt_per_class[c]),
            **dict(zip(('precision','recall','map50','map95'),map(float,validator.metrics.seg.class_aware_results(i)))))
            for i,c in enumerate(validator.metrics.seg.ap_class_idx)},validation_options=validation_options)
    del validator
    gc.collect();torch.cuda.empty_cache()
    result['forward_benchmark']=benchmark_forward(checkpoint)
    assert before==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result['checkpoint_unchanged']=True
    return result
