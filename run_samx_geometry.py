"""Run only the new P4 geometric residual trial with durable status and checks."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

from samx_experiment_config import (GEOMETRY_AUDIT as AUDIT, ROOT,
                                    clear_validation_cache, training_options)
from run_samx_experiments import now, write_json


SOURCES = [
    'vajra/nn/samx_geometry.py', 'vajra/nn/samx.py', 'vajra/nn/modules.py',
    'vajra/nn/vajra.py', 'vajra/nn/head.py', 'vajra/core/trainer.py',
    'vajra/core/model.py', 'vajra/loss/__init__.py',
    'vajra/models/vajra/segment/train.py', 'vajra/configs/__init__.py',
    'vajra/configs/hyperparams/default_hyp.yaml', 'samx_experiment_config.py',
    'seg_train.py', 'run_samx_geometry.py', 'run_samx_experiments.py',
    'tests/test_samx_geometry.py', 'tests/test_samx.py',
]


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def run(smoke=False):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv
    from vajra.nn.samx_geometry import GeometricDifference, GeometryResidualSAMX, ResidualScale

    torch.set_num_threads(4)
    options = training_options('geometry_residual', smoke=smoke)
    prefix = 'smoke' if smoke else 'train'
    progress_path = AUDIT/(prefix+'_progress.json')
    AUDIT.mkdir(parents=True, exist_ok=True)
    run_dir = Path(options['project'])/options['name']
    if progress_path.exists() or run_dir.exists():
        raise FileExistsError(f'Refusing duplicate run or renamed output: {progress_path}, {run_dir}')
    source_manifest = hashes()
    write_json(AUDIT/(prefix+'_source_manifest.json'), source_manifest)
    with zipfile.ZipFile(AUDIT/(prefix+'_source.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT/name, name)
    old_result = ROOT/'output'/'samx_experiments_20261027'/'local_global_result.json'
    old_options = json.loads(old_result.read_text(encoding='utf-8'))['options']
    diff = {k:dict(previous=old_options.get(k), current=options.get(k))
            for k in old_options.keys() | options.keys() if old_options.get(k)!=options.get(k)}
    if not smoke:
        assert set(diff) == {'name','samx_mode','samx_geometry_gain'}, diff
    write_json(AUDIT/(prefix+'_configuration_difference.json'), diff)
    record = dict(status='initializing', mode='geometry_residual', smoke=smoke,
                  pid=os.getpid(), started_at=now(), updated_at=now(), options=options,
                  original_control_queued=False, p3_trial_queued=False,
                  comparison_note='Shared-trunk geometric residual is a complete variant, not an isolated gate ablation.')
    write_json(progress_path, record)
    started = time.monotonic()
    model = Vajra('vajra-v1-nano-seg')
    captured = {}
    counters = dict(batches_seen=0, last_update=0.)
    trace_stream = (AUDIT/(prefix+'_validation_trace.log')).open('a',encoding='utf-8')

    def validation_start(validator):
        # Release unused training allocations before a larger inference batch.
        # On Windows the allocator can otherwise spill cached pages to shared RAM.
        clear_validation_cache(validator)
        faulthandler.dump_traceback_later(60,repeat=True,file=trace_stream)
        counters['validation_started']=time.monotonic()
        record.update(status='validating_training_epoch',updated_at=now())
        write_json(progress_path,record)
        print('GEOMETRY_VALIDATION_START '+json.dumps(dict(
            batch_size=validator.dataloader.batch_size,
            allocated_gib=torch.cuda.memory_allocated()/2**30,
            reserved_gib=torch.cuda.memory_reserved()/2**30)),flush=True)

    def validation_batch_start(validator):
        counters['validation_batch_started']=time.monotonic()

    def validation_batch_end(validator):
        if validator.batch_i<4:
            print('GEOMETRY_VALIDATION_BATCH '+json.dumps(dict(
                index=validator.batch_i,seconds=time.monotonic()-counters['validation_batch_started'],
                allocated_gib=torch.cuda.memory_allocated()/2**30,
                reserved_gib=torch.cuda.memory_reserved()/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)

    def validation_end(validator):
        faulthandler.cancel_dump_traceback_later()

    def gains(trainer):
        return {name:float(m.weight.detach().float().cpu())
                for name,m in trainer.model.named_modules() if isinstance(m,ResidualScale)}

    def verify_start(trainer):
        neck = trainer.model.model[0].vajra_neck3
        assert type(neck) is GeometryResidualSAMX, type(neck)
        head = trainer.model.model[-1]
        assert isinstance(head.proto_enhance,nn.Identity)
        assert isinstance(head.seg_enhance,nn.Identity)
        assert not head.use_cfsav1 and not head.use_fft_guided_ema and not head.use_fft_guided_ema_proto_only
        for key in ('use_cfsa','use_cfsav1','cfsav1_global_context','cfsav1_semantic_context',
                    'cfsav1_gate_smooth','cfsav1_gate_ramp_epochs','cfsav1_warmup_epochs'):
            assert not getattr(trainer.args,key), (key,getattr(trainer.args,key))
        assert not trainer.model.init_criterion().use_morph_mask_loss
        assert trainer.amp, 'AMP disabled; abort rather than silently alter settings'
        geometries = [m for m in trainer.model.modules() if isinstance(m,GeometricDifference)]
        assert len(geometries)==2
        assert len([m for m in trainer.model.modules() if isinstance(m,LDConv)])==2
        for m in geometries:
            assert m.p_conv.weight.count_nonzero().item()==0
            assert m.p_conv.bias.count_nonzero().item()==0
        offsets = {id(p) for m in geometries for p in m.p_conv.parameters()}
        scales = [m.weight for m in trainer.model.modules() if isinstance(m,ResidualScale)]
        assert len(offsets)==4 and len(scales)==2
        assert all(abs(p.item()-.1)<1e-6 for p in scales)
        ids=[id(p) for g in trainer.optimizer.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))
        # Framework also groups the frozen DFL convolution; it has no gradient.
        assert set(ids)=={id(p) for p in trainer.model.parameters()}
        assert all(ids.count(id(p))==1 for p in scales)
        assert {id(p) for g in trainer.optimizer.param_groups if g.get('lr_scale')==.1 for p in g['params']}==offsets
        train_count=len(trainer.train_loader.dataset)
        assert train_count==(61 if smoke else 6134), train_count
        assert len(trainer.test_loader.dataset)==1532
        captured.update(neck_class=type(neck).__name__, insertion='P4 / vajra_neck3',
                        offset_parameter_tensors=4, gain_parameter_tensors=2,
                        initial_gains=gains(trainer), train_images=train_count, val_images=1532,
                        parameters=sum(p.numel() for p in trainer.model.parameters()),
                        cfsa_enabled=False,cfsav1_enabled=False,mabd_enabled=False,
                        amp=bool(trainer.amp),run_dir=str(trainer.save_dir),
                        optimizer=type(trainer.optimizer).__name__,
                        optimizer_groups=[dict(lr=g['lr'],decay=g['weight_decay'],
                            lr_scale=g.get('lr_scale',1),warmup_bias=g.get('warmup_bias',i==0),
                            tensors=len(g['params'])) for i,g in enumerate(trainer.optimizer.param_groups)])
        assert str(trainer.save_dir)==str(run_dir), (trainer.save_dir,run_dir)
        record.update(status='training',startup=captured,updated_at=now())
        write_json(AUDIT/(prefix+'_startup.json'),captured)
        write_json(progress_path,record)
        print('GEOMETRY_STARTUP_VERIFIED '+json.dumps(captured),flush=True)

    def batch_done(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError('Non-finite training loss')
        for g in trainer.optimizer.param_groups:
            if g.get('lr_scale')!=.1:
                continue
            matching=[b for i,b in enumerate(trainer.optimizer.param_groups)
                      if 'lr_scale' not in b and b['weight_decay']==g['weight_decay']
                      and (i==0)==g['warmup_bias']]
            assert matching and abs(g['lr']-.1*matching[0]['lr'])<1e-10
        counters['batches_seen']+=1
        if not captured.get('first_batch_verified'):
            captured.update(first_batch_verified=True,warmup_ratio_verified=True,
                            batch_size=trainer.batch_size,
                            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(AUDIT/(prefix+'_startup.json'),captured)
            print('GEOMETRY_FIRST_BATCH_VERIFIED '+json.dumps(captured),flush=True)
        current=time.monotonic()
        if current-counters['last_update']>10:
            counters['last_update']=current
            record.update(status='training',updated_at=now(),current_epoch=trainer.epoch+1,
                          batches_seen=counters['batches_seen'],loss=float(trainer.loss.detach()),
                          elapsed_seconds=current-started,
                          gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(progress_path,record)

    def epoch_done(trainer):
        record.update(status='training',updated_at=now(),last_completed_epoch=trainer.epoch+1,
                      best_epoch=int(trainer.stopper.best_epoch),
                      best_fitness=float(trainer.best_fitness) if trainer.best_fitness is not None else None,
                      metrics={k:float(v) for k,v in (trainer.metrics or {}).items()},
                      gains=gains(trainer),elapsed_seconds=time.monotonic()-started,
                      gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(progress_path,record)
        print('GEOMETRY_EPOCH_COMPLETE '+json.dumps(record),flush=True)

    model.add_callback('on_pretrain_routine_end',verify_start)
    model.add_callback('on_train_batch_end',batch_done)
    model.add_callback('on_fit_epoch_end',epoch_done)
    model.add_callback('on_val_start',validation_start)
    model.add_callback('on_val_batch_start',validation_batch_start)
    model.add_callback('on_val_batch_end',validation_batch_end)
    model.add_callback('on_val_end',validation_end)
    model.train(**options)
    trainer=model.trainer
    checkpoint=Path(trainer.best)
    assert checkpoint.exists(), checkpoint
    record.update(status='validating_fp32',updated_at=now(),checkpoint=str(checkpoint))
    write_json(progress_path,record)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    val_options=dict(data=options['data'],split='val',img_size=1600,rect=True,
        batch=8,workers=0,device=0,half=False,conf=.001,iou=.7,max_det=300,
        overlap_mask=True,mask_ratio=4,augment=False,plots=False,verbose=False,
        project=str(AUDIT/'validation'),name=prefix,exist_ok=False)
    validated=Vajra(str(checkpoint))
    counts={}
    def count_validation(validator):
        counts.update(images=int(validator.seen),instances=int(validator.nt_per_class.sum()))
    validated.add_callback('on_val_end',count_validation)
    metrics=validated.val(**val_options)
    assert counts==dict(images=1532,instances=3942), counts
    assert checkpoint_hash==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    final_hashes=hashes()
    source_changes={k:dict(start=v,end=final_hashes[k]) for k,v in source_manifest.items()
                    if final_hashes[k]!=v}
    result=dict(mode='geometry_residual',smoke=smoke,finished_at=now(),checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash,options=options,validation_options=val_options,
                startup=captured,validation_counts=counts,completed_epochs=int(trainer.epoch)+1,
                best_epoch=int(trainer.stopper.best_epoch),elapsed_seconds=time.monotonic()-started,
                source_unchanged=not source_changes,source_changes=source_changes,
                metrics={k:float(v) for k,v in metrics.results_dict.items()},
                speed={k:float(v) for k,v in metrics.speed.items()})
    write_json(AUDIT/(prefix+'_result.json'),result)
    if not smoke:
        results=[json.loads((ROOT/'output'/'samx_experiments_20261027'/(m+'_result.json')).read_text(encoding='utf-8'))
                 for m in ('stage_global','local_global')]+[result]
        keys=['metrics/precision(Mask)','metrics/recall(Mask)','metrics/mAP50(Mask)','metrics/mAP50-95(Mask)']
        write_json(AUDIT/'comparison.json',dict(percentage_values=[dict(mode=r['mode'],**{k:r['metrics'][k]*100 for k in keys}) for r in results],
            historical_paper_samx=dict(precision=65.80,recall=58.60,map50=61.30,map50_95=27.60),
            note='Paper row is historical, not a new repaired-original control. One seed; no significance claim.'))
    record.update(status='complete',updated_at=now(),result_file=str(AUDIT/(prefix+'_result.json')))
    write_json(progress_path,record)
    print('GEOMETRY_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    if args.smoke==args.train:
        parser.error('Choose exactly one of --smoke / --train')
    prefix='smoke' if args.smoke else 'train'
    try:
        run(smoke=args.smoke)
    except BaseException:
        error=dict(status='failed',updated_at=now(),pid=os.getpid(),traceback=traceback.format_exc())
        progress_path=AUDIT/(prefix+'_progress.json')
        if progress_path.exists():
            try:
                previous=json.loads(progress_path.read_text(encoding='utf-8'))
                previous.update(error)
                error=previous
            except (OSError,json.JSONDecodeError):
                pass
        write_json(AUDIT/(prefix+'_failure.json'),error)
        write_json(progress_path,error)
        raise
