"""Training-only weight calibration and fixed-threshold validation error cases."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mabd_followup_config import AUDIT, previous_training_options, bounded_gain, validation_options, SCALE_MISMATCH
from run_samx_experiments import now, write_json


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()),
        **dict(zip(('p05', 'p25', 'p50', 'p75', 'p95'), map(float, np.quantile(values, [.05,.25,.5,.75,.95])))))


def training_statistics():
    from vajra.configs import get_config
    from vajra.dataset import build_vajra_dataset, build_dataloader
    from vajra.dataset.utils import check_det_dataset
    from vajra.loss import SegmentationLoss
    from vajra.loss.mabd_v1 import erode_zero_padded
    from vajra.ops import xywh_to_xyxy, crop_mask
    from vajra.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    cfg = get_config(model_configuration={**previous_training_options(), 'task':'segment','mode':'train'})
    data = check_det_dataset(cfg.data)
    dataset = build_vajra_dataset(cfg, data['train'], 32, data, mode='train', stride=32)
    assert len(dataset) == 6134 and dataset.rect
    loader = build_dataloader(dataset, 32, 0, shuffle=False)
    columns = {key: [] for key in ('class_id','area','thinness','fill_factor','shape_factor')}
    seen = []
    label_digest = hashlib.sha256()
    for label in dataset.label_files:
        path = Path(label)
        label_digest.update(str(path).encode('utf-8'))
        label_digest.update(path.read_bytes() if path.exists() else b'MISSING')
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            h, w = batch['masks'].shape[-2:]
            for si, filename in enumerate(batch['im_file']):
                select = batch['batch_idx'] == si
                classes = batch['cls'][select].reshape(-1).long()
                seen.append(filename)
                if not len(classes):
                    continue
                gt = (batch['masks'][si:si+1] == torch.arange(1,len(classes)+1)[:,None,None]).float()
                boxes = xywh_to_xyxy(batch['bboxes'][select]) * torch.tensor([w,h,w,h])
                crop_area = ((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])).clamp_min(1e-6)
                shape, fill = SegmentationLoss.morphology_factors(gt, boxes, crop_area)
                area = gt.sum((1,2))
                q = ((gt-erode_zero_padded(gt)).clamp(0,1).sum((1,2))/area.clamp_min(1e-6)).clamp(0,1)
                for key, values in dict(class_id=classes, area=area, thinness=q, fill_factor=fill, shape_factor=shape).items():
                    columns[key].extend(values.tolist())
            if bi % 20 == 0:
                write_json(AUDIT/'diagnostics_progress.json',dict(phase='training_weight_statistics',
                    images=len(seen), unique_instances=len(columns['area']), updated_at=now()))
    assert len(seen) == len(set(seen)) == 6134
    arrays = {key: np.asarray(value) for key,value in columns.items()}
    np.savez_compressed(AUDIT/'training_instance_stats.npz', **arrays)
    valid = arrays['area'] > 0
    q, old = arrays['thinness'][valid], .15*arrays['fill_factor'][valid]
    e1, current = .15*q, .05+.15*q
    target = float(old.mean())
    calibration = dict(target_original_mean=target, e1_mean=float(e1.mean()),
        relative_mean_mismatch=float(abs(e1.mean()-target)/max(target,1e-12)),
        cap=.15, fitted_on='one augmented training pass, nonempty unique GT masks only',
        validation_or_test_used=False, eligible=False)
    try:
        calibration['gain'] = bounded_gain(q,target)
        calibration['achieved_mean'] = float(np.minimum(calibration['gain']*q,.15).mean())
        calibration['eligible'] = calibration['relative_mean_mismatch'] >= SCALE_MISMATCH
    except ValueError as exc:
        calibration['reason'] = str(exc)
    weights = dict(original=.15*arrays['fill_factor'], current=.05+.15*arrays['thinness'], e1=.15*arrays['thinness'])
    def group(select):
        return dict(instances=int(select.sum()), empty_masks=int((arrays['area'][select]==0).sum()),
            thinness=describe(arrays['thinness'][select]),
            weights={key:describe(value[select]) for key,value in weights.items()})
    summary = dict(created_at=now(), images=6134, unique_instances=len(valid), empty_masks=int((~valid).sum()),
        protocol='Exactly one GT observation per augmented image/instance; no positive-anchor duplication. '
                 'seed42, actual train transforms, rect=True, 1600, mask_ratio4, overlap masks; '
                 'one pass is a diagnostic sample, not all epoch distributions.',
        data_yaml_sha256=hashlib.sha256(Path(cfg.data).read_bytes()).hexdigest(),
        label_content_sha256=label_digest.hexdigest(),
        image_list_sha256=hashlib.sha256('\n'.join(seen).encode('utf-8')).hexdigest(),
        overall=group(np.ones(len(valid),dtype=bool)),
        per_class={str(int(c)):group(arrays['class_id']==c) for c in np.unique(arrays['class_id'])},
        per_size={'small_le16_proto_pixels':group(arrays['area']<=16),
                  'medium_17_to256':group((arrays['area']>16)&(arrays['area']<=256)),
                  'large_gt256':group(arrays['area']>256)},
        per_thinness={'high_ge075':group(arrays['thinness']>=.75),
                      'middle_025_to075':group((arrays['thinness']>=.25)&(arrays['thinness']<.75)),
                      'low_lt025':group(arrays['thinness']<.25)}, calibration=calibration,
        caveat='Thinness mixes instance size with shape. Mean weight calibration does not match gradient magnitudes.')
    write_json(AUDIT/'weight_diagnostics.json',summary)
    del loader,dataset
    gc.collect()
    return summary


def error_diagnostics(references):
    """Reuse fixed-conf counts; save visual cases for original and latest loss."""
    from samx_followup_validation import BoundarySegmentationValidator
    from samx_experiment_config import clear_validation_cache
    from vajra.metrics import mask_iou
    from vajra.utils.torch_utils import init_seeds
    import cv2

    class CaseValidator(BoundarySegmentationValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.cases, self.visuals = [], []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds,batch)
            for si,(pred,proto) in enumerate(zip(preds[0],preds[1])):
                prepared=self._prepare_batch(si,batch)
                cls=prepared['cls']
                gt=prepared['masks']==torch.arange(1,len(cls)+1,device=self.device)[:,None,None]
                selected=pred[pred[:,4]>=.25]
                pm=self._prepare_pred(selected,prepared,proto)[1] if len(selected) else gt.new_zeros(0,*gt.shape[-2:])
                if len(gt) and gt.shape[-2:]!=pm.shape[-2:]:
                    gt=F.interpolate(gt[None].float(),pm.shape[-2:],mode='bilinear',align_corners=False)[0]>.5
                used_g,used_p,pairs=set(),set(),[]
                if len(gt) and len(pm):
                    matrix=mask_iou(gt.flatten(1).float(),pm.flatten(1).float())
                    indices=((matrix>=.5)&(cls[:,None]==selected[:,5][None])).nonzero().cpu().tolist()
                    vals=matrix.cpu()
                    for g,p in sorted(indices,key=lambda pair:(-float(vals[pair[0],pair[1]]),*pair)):
                        if g not in used_g and p not in used_p:
                            used_g.add(g);used_p.add(p)
                            ga,pa=gt[g].cpu().numpy().astype(np.uint8),pm[p].cpu().numpy().astype(np.uint8)
                            missing=float(((ga>0)&(pa==0)).sum()/max(ga.sum(),1))
                            extra=float(((pa>0)&(ga==0)).sum()/max(ga.sum(),1))
                            fragmentation=max(0,cv2.connectedComponents(pa,connectivity=8)[0]-cv2.connectedComponents(ga,connectivity=8)[0])
                            pairs.append(dict(gt_id=g,pred_id=p,mask_iou=float(vals[g,p]),
                                missing_foreground_fraction=missing,extra_background_per_gt_area=extra,
                                extra_connected_components=fragmentation))
                fp,fn=len(pm)-len(used_p),len(gt)-len(used_g)
                row=dict(image=batch['im_file'][si],gt_instances=len(gt),pred_instances=len(pm),
                    tp=len(pairs),fp=fp,fn=fn,matched_details=pairs)
                self.cases.append(row)
                severity=fp+fn+sum(p['missing_foreground_fraction']+p['extra_background_per_gt_area'] for p in pairs)
                if severity>0 and (len(self.visuals)<12 or severity>self.visuals[-1][0]):
                    image=(batch['img'][si].detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
                    panels=[]
                    for masks,color,title in ((gt,(40,210,40),'GT'),(pm,(230,60,40),'Prediction conf>=0.25')):
                        panel=image.copy()
                        if len(masks):
                            union=cv2.resize(masks.any(0).cpu().numpy().astype(np.uint8),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                            panel[union]=(panel[union]*.4+np.asarray(color)*.6).astype(np.uint8)
                        cv2.putText(panel,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        panels.append(panel)
                    combined=np.concatenate(panels,axis=0)
                    self.visuals.append((severity,row,combined))
                    self.visuals.sort(key=lambda item:-item[0]);self.visuals=self.visuals[:12]

    summary={}
    for label,result in references.items():
        counts=result['boundary']['overall']
        tp=counts['matched_instances'];fp=counts['pred_instances']-tp;fn=counts['gt_instances']-tp
        summary[label]=dict(tp=tp,fp=fp,fn=fn,confidence=.25,mask_iou=.5,
            precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
    for label in ('original_MABD','MABDv1_original_dice_thinness_boundary'):
        init_seeds(42,deterministic=True)
        result=references[label]
        before=hashlib.sha256(Path(result['checkpoint']).read_bytes()).hexdigest()
        validator=CaseValidator(args={**validation_options('diagnostic_'+label),'task':'segment','mode':'val'})
        validator.add_callback('on_val_start',clear_validation_cache)
        validator(model=result['checkpoint'])
        assert validator.seen==1532
        case_dir=AUDIT/'error_cases'/label
        case_dir.mkdir(parents=True,exist_ok=False)
        write_json(case_dir/'all_images.json',validator.cases)
        listing=[]
        for i,(severity,row,visual) in enumerate(validator.visuals):
            path=case_dir/f'{i+1:02d}.jpg'
            cv2.imencode('.jpg',cv2.cvtColor(visual,cv2.COLOR_RGB2BGR))[1].tofile(str(path))
            listing.append(dict(**row,severity=severity,visual=str(path)))
        write_json(case_dir/'top_cases.json',listing)
        assert before==hashlib.sha256(Path(result['checkpoint']).read_bytes()).hexdigest()
        summary[label]['matched_missing_foreground_mean']=float(np.mean([
            pair['missing_foreground_fraction'] for row in validator.cases for pair in row['matched_details']]))
        del validator
        gc.collect();torch.cuda.empty_cache()
    write_json(AUDIT/'fixed_threshold_errors.json',summary)
    return summary


def run_diagnostics(references):
    summary=training_statistics()
    error_diagnostics(references)
    calibration=summary['calibration']
    lines=['# MABD 权重诊断','',f"训练图像：{summary['images']}；唯一实例观测：{summary['unique_instances']}。",'',
           '按实际训练增强进行固定种子的一次遍历；校准只使用训练集非空 GT。', '',
           '| 版本 | 平均边界权重 |','| --- | ---: |']
    for label,weight in summary['overall']['weights'].items():
        lines.append(f"| {label} | {weight['mean']:.6f} |")
    lines+=['',f"E1 与原权重的相对均值差：{calibration['relative_mean_mismatch']:.2%}。",
            f"条件 E2 的训练集校准系数：{calibration.get('gain')}；满足强度失配条件：{calibration['eligible']}。",'',
            '固定阈值误检/漏检见 fixed_threshold_errors.json；可视化见 error_cases/。',
            '均值失配只能支持安排校准实验，不能证明它是精度下降的原因。']
    (AUDIT/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
