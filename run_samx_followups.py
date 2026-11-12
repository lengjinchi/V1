"""Preflight then sequential training: dual P4 -> free Proto -> structured Proto."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from run_samx_experiments import now,write_json
from samx_followup_config import AUDIT,BASELINE_RESULTS,MODES,ROOT,options


SOURCES=['vajra/nn/samx_followups.py','vajra/nn/samx.py','vajra/nn/samx_geometry.py',
         'vajra/nn/modules.py','vajra/nn/head.py','vajra/nn/vajra.py','vajra/core/trainer.py',
         'vajra/core/model.py','vajra/models/vajra/segment/train.py','vajra/loss/__init__.py',
         'vajra/configs/__init__.py','vajra/configs/hyperparams/default_hyp.yaml',
         'samx_experiment_config.py','samx_followup_config.py','samx_followup_validation.py',
         'run_samx_followups.py','run_samx_experiments.py','tests/test_samx_followups.py',
         'samx_followup_recovery.py','tests/test_samx_followup_recovery.py']


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def check_protected():
    records=json.loads((AUDIT/'protected_existing_weights.json').read_text(encoding='utf-8'))
    for r in records:
        p=ROOT/r['path']
        assert p.exists() and p.stat().st_size==r['size'] and p.stat().st_mtime_ns==r['mtime_ns'],r['path']
        if 'sha256' in r:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
    return dict(existing_checkpoints_preserved=len(records),checked_at=now())


def worker(mode,smoke=False,resume=False):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv
    from vajra.nn.samx import LocalGlobalSAMX
    from vajra.nn.samx_geometry import ResidualScale
    from vajra.nn.samx_followups import (DualPathGeometrySAMX,ProtoGeometryEnhance,
                                        StructuredGeometricDifference)
    from samx_experiment_config import clear_validation_cache
    from samx_followup_validation import validate_checkpoint
    from samx_followup_recovery import FollowupResumeTrainer

    torch.set_num_threads(4)
    args=options(mode,smoke)
    prefix=('smoke_' if smoke else '')+mode
    progress_path=AUDIT/(prefix+'_progress.json')
    run_dir=Path(args['project'])/args['name']
    if not resume and (progress_path.exists() or run_dir.exists()):
        raise FileExistsError(f'Refusing duplicate/renamed experiment: {run_dir}')
    recovery=None
    if resume:
        assert mode=='proto_structured' and not smoke
        recovery=json.loads((AUDIT/'resume_context.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(Path(recovery['checkpoint']).read_bytes()).hexdigest()==recovery['checkpoint_sha256']
    manifest=source_hashes()
    write_json(AUDIT/(prefix+('_resume' if resume else '')+'_source_manifest.json'),manifest)
    state=dict(status='initializing',mode=mode,smoke=smoke,pid=os.getpid(),started_at=now(),
               updated_at=now(),options=args,last_completed_epoch=0)
    if resume:
        state.update(started_at=recovery['original_started_at'],resumed_at=now(),
                     last_completed_epoch=recovery['completed_epochs'],recovery=recovery)
    write_json(progress_path,state)
    started=time.monotonic();batch_count=[0];last_update=[0.]
    prior_elapsed=recovery['elapsed_seconds'] if resume else 0.
    model=Vajra(recovery['checkpoint'] if resume else 'vajra-v1-nano-seg')
    startup={}
    def gain_values(trainer):
        return {n:float(m.weight.detach().float().cpu()) for n,m in trainer.model.named_modules()
                if isinstance(m,ResidualScale)}
    def verify_start(trainer):
        neck,head=trainer.model.model[0].vajra_neck3,trainer.model.model[-1]
        assert type(neck) is (DualPathGeometrySAMX if mode=='dual_geometry' else LocalGlobalSAMX)
        assert isinstance(head.seg_enhance,nn.Identity)
        if mode=='dual_geometry':
            assert isinstance(head.proto_enhance,nn.Identity)
        else:
            assert type(head.proto_enhance) is ProtoGeometryEnhance
            assert head.proto_enhance.mode==args['samx_proto_mode']
            assert head.proto_enhance.hidden==16
        for flag in ('use_cfsa','use_cfsav1','cfsav1_global_context','cfsav1_semantic_context',
                     'cfsav1_gate_smooth','cfsav1_gate_ramp_epochs','cfsav1_warmup_epochs'):
            assert not getattr(trainer.args,flag),(flag,getattr(trainer.args,flag))
        assert not head.use_cfsav1 and not head.use_fft_guided_ema and not head.use_fft_guided_ema_proto_only
        assert not trainer.model.init_criterion().use_morph_mask_loss
        assert trainer.amp,'AMP disabled unexpectedly'
        samplers=[m for m in trainer.model.modules() if isinstance(m,LDConv)]
        assert len(samplers)==(2 if mode=='dual_geometry' else 3)
        assert sum(isinstance(m,StructuredGeometricDifference) for m in samplers)==int(mode=='proto_structured')
        if not resume:
            for m in samplers:
                assert m.p_conv.weight.count_nonzero().item()==0 and m.p_conv.bias.count_nonzero().item()==0
        offsets={id(p) for m in samplers for p in m.p_conv.parameters()}
        ids=[id(p) for g in trainer.optimizer.param_groups for p in g['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in trainer.model.parameters()}
        assert {id(p) for g in trainer.optimizer.param_groups if g.get('lr_scale')==.1 for p in g['params']}==offsets
        gains=gain_values(trainer)
        assert len(gains)==(2 if mode=='dual_geometry' else 1)
        if not resume:
            assert all(abs(v-.1)<1e-6 for v in gains.values())
        assert all(ids.count(id(m.weight))==1 for m in trainer.model.modules() if isinstance(m,ResidualScale))
        assert len(trainer.train_loader.dataset)==(61 if smoke else 6134)
        assert len(trainer.test_loader.dataset)==1532
        assert Path(trainer.save_dir)==run_dir
        if resume:
            trainer.verify_recovery(recovery)
        startup.update(neck_class=type(neck).__name__,proto_class=type(head.proto_enhance).__name__,
            samplers=[type(m).__name__ for m in samplers],offset_parameter_tensors=len(offsets),
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            train_images=len(trainer.train_loader.dataset),val_images=1532,
            cfsa_enabled=False,cfsav1_enabled=False,mabd_enabled=False,amp=bool(trainer.amp),
            optimizer=type(trainer.optimizer).__name__,batch_size=trainer.batch_size,
            optimizer_groups=[dict(lr=g['lr'],decay=g['weight_decay'],lr_scale=g.get('lr_scale',1),
                warmup_bias=g.get('warmup_bias',i==0),tensors=len(g['params']))
                for i,g in enumerate(trainer.optimizer.param_groups)])
        startup['resumed_gains' if resume else 'initial_gains']=gains
        if resume:
            startup.update(resumed=True,start_epoch=trainer.start_epoch+1,
                restored_ema_updates=trainer.ema.updates,
                restored_best_epoch=trainer.stopper.best_epoch,
                restored_best_fitness=trainer.best_fitness,
                student_optimizer_ema_verified=True,
                resume_limitations='Original checkpoints store half-precision model/EMA and no RNG or AMP scaler state; not a bitwise continuation.')
        state.update(status='training',startup=startup,updated_at=now())
        write_json(AUDIT/(prefix+('_resume' if resume else '')+'_startup.json'),startup)
        write_json(progress_path,state,best_effort=True)
        print('FOLLOWUP_STARTUP_VERIFIED '+json.dumps(startup),flush=True)
    def batch_done(trainer):
        if not torch.isfinite(trainer.loss).all():raise FloatingPointError('Non-finite loss')
        for g in trainer.optimizer.param_groups:
            if g.get('lr_scale')!=.1:continue
            base=[b for i,b in enumerate(trainer.optimizer.param_groups) if 'lr_scale' not in b
                  and b['weight_decay']==g['weight_decay'] and (i==0)==g['warmup_bias']]
            assert base and abs(g['lr']-.1*base[0]['lr'])<1e-10
        batch_count[0]+=1
        if not startup.get('first_batch_verified'):
            startup.update(first_batch_verified=True,warmup_ratio_verified=True,
                           gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(AUDIT/(prefix+('_resume' if resume else '')+'_startup.json'),startup)
        if time.monotonic()-last_update[0]>10:
            last_update[0]=time.monotonic()
            state.update(status='training',updated_at=now(),current_epoch=trainer.epoch+1,
                batches_seen=batch_count[0],loss=float(trainer.loss.detach()),
                elapsed_seconds=prior_elapsed+time.monotonic()-started,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(progress_path,state,best_effort=True)
    def val_start(validator):
        clear_validation_cache(validator)
        state.update(status='validating_epoch',updated_at=now())
        write_json(progress_path,state,best_effort=True)
    def epoch_done(trainer):
        state.update(status='training',updated_at=now(),last_completed_epoch=trainer.epoch+1,
            best_epoch=int(trainer.stopper.best_epoch),best_fitness=float(trainer.best_fitness),
            metrics={k:float(v) for k,v in (trainer.metrics or {}).items()},gains=gain_values(trainer),
            elapsed_seconds=prior_elapsed+time.monotonic()-started,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(progress_path,state,best_effort=True)
        print('FOLLOWUP_EPOCH_COMPLETE '+json.dumps(state),flush=True)
    model.add_callback('on_pretrain_routine_end',verify_start)
    model.add_callback('on_train_batch_end',batch_done)
    model.add_callback('on_val_start',val_start)
    model.add_callback('on_fit_epoch_end',epoch_done)
    model.train(**({**args,'resume':True,'trainer':FollowupResumeTrainer} if resume else args))
    trainer=model.trainer
    assert Path(trainer.best).exists()
    state.update(status='validating_fp32_boundary_speed',updated_at=now())
    write_json(progress_path,state)
    val=dict(data=args['data'],split='val',img_size=1600,rect=True,batch=8,workers=0,device=0,
        half=False,conf=.001,iou=.7,max_det=300,overlap_mask=True,mask_ratio=4,
        augment=False,plots=False,verbose=False,project=str(AUDIT/'validation'),name=prefix,exist_ok=False)
    result=validate_checkpoint(trainer.best,val)
    result.update(mode=mode,smoke=smoke,options=args,startup=startup,finished_at=now(),
        completed_epochs=int(trainer.epoch)+1,best_epoch=int(trainer.stopper.best_epoch),
        elapsed_seconds=prior_elapsed+time.monotonic()-started,source_unchanged=source_hashes()==manifest,
        training_peak_allocated_gib=state['gpu_peak_allocated_gib'])
    if resume:
        result.update(resumed=True,recovery=recovery,
                      source_unchanged_scope='Since operational repair/resume; model sources match original queue snapshot.')
    write_json(AUDIT/(prefix+'_result.json'),result)
    state.update(status='complete',updated_at=now(),result_file=str(AUDIT/(prefix+'_result.json')))
    write_json(progress_path,state)
    print('FOLLOWUP_COMPLETE '+json.dumps(result),flush=True)


def baseline_worker(label):
    import torch
    from samx_followup_validation import validate_checkpoint
    torch.set_num_threads(4)
    old=json.loads(BASELINE_RESULTS[label].read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(old['checkpoint']).read_bytes()).hexdigest()==old['checkpoint_sha256']
    val={**old['validation_options'],'project':str(AUDIT/'baseline_validation'),'name':label,'exist_ok':False}
    result=validate_checkpoint(old['checkpoint'],val,expected=old['metrics'])
    result.update(label=label,previous_result=str(BASELINE_RESULTS[label]),baseline_reproduced=True,finished_at=now())
    write_json(AUDIT/('baseline_'+label+'_result.json'),result)
    print('BASELINE_REPRODUCED '+json.dumps(result),flush=True)


def run_child(arguments,logfile,state,status_file):
    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
    with logfile.open('x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*arguments],
                               cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
        state.pop('worker_progress',None)
        progress_path=None
        if '--worker' in arguments:
            label=arguments[arguments.index('--worker')+1]
            progress_path=AUDIT/(('smoke_' if '--smoke' in arguments else '')+label+'_progress.json')
        def refresh_progress():
            if progress_path is not None and progress_path.exists():
                try:state['worker_progress']=json.loads(progress_path.read_text(encoding='utf-8'))
                except (OSError,json.JSONDecodeError):pass
        state.update(worker_pid=child.pid,updated_at=now());write_json(status_file,state)
        while child.poll() is None:
            refresh_progress()
            state['updated_at']=now()
            write_json(status_file,state,best_effort=True)
            time.sleep(5)
        refresh_progress()
        write_json(status_file,state)
        if child.returncode:
            raise RuntimeError(f'Child {arguments} exited {child.returncode}; inspect {logfile}')


def preflight():
    status_file=AUDIT/'preflight_status.json'
    if status_file.exists():raise FileExistsError('Preflight status exists; inspect before duplicate execution')
    tests=json.loads((AUDIT/'tests_result.json').read_text(encoding='utf-8'))
    assert tests['success']
    manifest=source_hashes();write_json(AUDIT/'preflight_source_manifest.json',manifest)
    state=dict(status='running',pid=os.getpid(),started_at=now(),completed=[],phase='smoke')
    try:
        for mode in MODES:
            assert source_hashes()==manifest,'Source changed during preflight'
            state['current_mode']=mode
            run_child(['--worker',mode,'--smoke'],AUDIT/('smoke_'+mode+'.log'),state,status_file)
            state['completed'].append('smoke_'+mode)
        state['phase']='baseline_revalidation'
        for label in BASELINE_RESULTS:
            assert source_hashes()==manifest
            state['current_mode']=label
            run_child(['--baseline-worker',label],AUDIT/('baseline_'+label+'.log'),state,status_file)
            state['completed'].append('baseline_'+label)
        assert source_hashes()==manifest
        protected=check_protected()
        state.update(status='complete',updated_at=now(),current_mode=None,protected=protected)
        write_json(status_file,state)
        write_json(AUDIT/'preflight_complete.json',dict(state=state,source_manifest=manifest,tests=tests))
    except BaseException:
        state.update(status='failed',updated_at=now(),traceback=traceback.format_exc())
        write_json(status_file,state);raise


def update_summary():
    records={}
    for label in BASELINE_RESULTS:
        records[label]=json.loads((AUDIT/('baseline_'+label+'_result.json')).read_text(encoding='utf-8'))
    for mode in MODES:
        p=AUDIT/(mode+'_result.json')
        if p.exists():records[mode]=json.loads(p.read_text(encoding='utf-8'))
    pairs={'dual_geometry':'geometry_shared','proto_free':'local_global','proto_structured':'proto_free'}
    keys=['metrics/precision(Mask)','metrics/recall(Mask)','metrics/mAP50(Mask)','metrics/mAP50-95(Mask)']
    deltas={m:{k:100*(records[m]['metrics'][k]-records[b]['metrics'][k]) for k in keys}
            for m,b in pairs.items() if m in records and b in records}
    write_json(AUDIT/'comparison.json',dict(records=records,direct_comparison=pairs,deltas_percentage_points=deltas,
        note='Same FP32 validation and boundary/speed protocol. Scratch runs, single seed. Original paper row is historical and is not a rerun.'))
    lines=['# SAMX 后续实验结果','',
        '旧模型保留。以下均为最佳权重的完整 FP32 验证；边界指标与网络前向 FPS 使用本轮统一补充协议。', '',
        '| 模型 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 | 网络前向 FPS |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,r in records.items():
        values=[f"{100*r['metrics'][k]:.2f}%" for k in keys]
        lines.append('| '+label+' | '+' | '.join(values)+f" | {100*r['boundary']['overall']['f1']:.2f}% | {r['forward_benchmark']['fps']:.2f} |")
    lines+=['','直接比较：dual_geometry 对共享几何残差；proto_free 对 LocalGlobal；proto_structured 对 proto_free。',
        '边界 F1 包含未匹配实例，conf=0.25、同类 Mask IoU≥0.5、掩膜网格容差 1 像素。FPS 为 FP32、batch1、288×1632 纯前向，不与论文不同配置速度混比。',
        '原论文 SAMX✓/CFSA×/MABD× 的历史 AP50:95 为 27.60%，本轮不补跑修复原版对照。']
    (AUDIT/'结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def queue():
    status_file=AUDIT/'status.json'
    if status_file.exists():raise FileExistsError('Queue status exists; refusing duplicate training')
    pre=json.loads((AUDIT/'preflight_complete.json').read_text(encoding='utf-8'))
    assert pre['state']['status']=='complete' and pre['tests']['success']
    manifest=source_hashes()
    assert manifest==pre['source_manifest'],'Source changed after preflight; do not train unverified code'
    check_protected()
    write_json(AUDIT/'queue_source_manifest.json',manifest)
    with zipfile.ZipFile(AUDIT/'queue_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:z.write(ROOT/name,name)
    state=dict(status='running',pid=os.getpid(),started_at=now(),updated_at=now(),
               modes=list(MODES),completed=[],original_control_queued=False)
    try:
        update_summary()
        for mode in MODES:
            assert source_hashes()==manifest,'Source changed; stop before next process'
            check_protected()
            state['current_mode']=mode
            run_child(['--worker',mode],AUDIT/(mode+'.log'),state,status_file)
            result=json.loads((AUDIT/(mode+'_result.json')).read_text(encoding='utf-8'))
            assert result['source_unchanged'],'Source changed during worker; result retained, queue stopped'
            state['completed'].append(mode)
            update_summary()
        state.update(status='complete',updated_at=now(),current_mode=None,protected=check_protected())
        write_json(status_file,state)
    except BaseException:
        state.update(status='failed',updated_at=now(),traceback=traceback.format_exc())
        write_json(status_file,state);raise


def resume_queue():
    """Continue only the failed third trial; archive its prior state first."""
    import torch
    status_file=AUDIT/'status.json'
    old=json.loads(status_file.read_text(encoding='utf-8'))
    assert old['status']=='failed' and old['current_mode']=='proto_structured'
    assert old['completed']==['dual_geometry','proto_free']
    tests=json.loads((AUDIT/'recovery_tests.json').read_text(encoding='utf-8'))
    assert tests['success']
    original=json.loads((AUDIT/'queue_source_manifest.json').read_text(encoding='utf-8'))
    manifest=source_hashes()
    changed=[name for name in original if original[name]!=manifest[name]]
    assert set(changed)=={'run_samx_experiments.py','run_samx_followups.py'},changed
    check_protected()
    for mode in old['completed']:
        result=json.loads((AUDIT/(mode+'_result.json')).read_text(encoding='utf-8'))
        assert hashlib.sha256(Path(result['checkpoint']).read_bytes()).hexdigest()==result['checkpoint_sha256']
    args=options('proto_structured')
    run_dir=Path(args['project'])/args['name']
    last=run_dir/'weights/last-vajra-v1-nano-seg.pt'
    best=run_dir/'weights/best-vajra-v1-nano-seg.pt'
    checkpoint=torch.load(last,map_location='cpu',weights_only=False)
    best_checkpoint=torch.load(best,map_location='cpu',weights_only=False)
    assert checkpoint['optimizer'] is not None
    assert all(checkpoint['train_args'][key]==value for key,value in args.items())
    assert len(checkpoint['train_results']['epoch'])==checkpoint['epoch']+1
    assert len(list(csv.DictReader((run_dir/'results.csv').open(encoding='utf-8'))))==checkpoint['epoch']+1
    backup=AUDIT/('recovery_'+time.strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(exist_ok=False)
    files=[status_file,AUDIT/'proto_structured_progress.json',AUDIT/'proto_structured_startup.json',
           AUDIT/'proto_structured_source_manifest.json',run_dir/'args.yaml',run_dir/'results.csv',last,best]
    for path in files:
        shutil.copy2(path,backup/path.name)
    context=dict(checkpoint=str(last),checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
        backup=str(backup),completed_epochs=checkpoint['epoch']+1,best_epoch=best_checkpoint['epoch']+1,
        best_fitness=checkpoint['best_fitness'],elapsed_seconds=old['worker_progress']['elapsed_seconds'],
        original_started_at=old['worker_progress']['started_at'],prepared_at=now(),
        changed_sources=changed,model_sources_match_original=True,
        limitations='No saved RNG/AMP scaler; half-precision saved model/EMA. Resume is not bitwise identical.')
    write_json(AUDIT/'resume_context.json',context)
    with zipfile.ZipFile(backup/'recovery_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:
            z.write(ROOT/name,name)
    write_json(backup/'source_manifest.json',manifest)
    del checkpoint,best_checkpoint
    gc.collect()
    state=dict(status='running',pid=os.getpid(),started_at=old['started_at'],resumed_at=now(),
               updated_at=now(),modes=list(MODES),completed=list(old['completed']),
               original_control_queued=False,current_mode='proto_structured',recovery=context)
    try:
        run_child(['--worker','proto_structured','--resume'],backup/'proto_structured_resume.log',state,status_file)
        result=json.loads((AUDIT/'proto_structured_result.json').read_text(encoding='utf-8'))
        assert result['source_unchanged']
        state['completed'].append('proto_structured')
        update_summary()
        state.update(status='complete',updated_at=now(),current_mode=None,protected=check_protected())
        write_json(status_file,state)
    except BaseException:
        state.update(status='failed',updated_at=now(),traceback=traceback.format_exc())
        write_json(status_file,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true')
    group.add_argument('--queue',action='store_true')
    group.add_argument('--resume-queue',action='store_true')
    group.add_argument('--worker',choices=MODES)
    group.add_argument('--baseline-worker',choices=list(BASELINE_RESULTS))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.smoke and not args.worker:parser.error('--smoke requires --worker')
    if args.resume and (args.worker!='proto_structured' or args.smoke):parser.error('--resume requires the formal proto_structured worker')
    try:
        if args.preflight:preflight()
        elif args.queue:queue()
        elif args.resume_queue:resume_queue()
        elif args.worker:worker(args.worker,args.smoke,args.resume)
        else:baseline_worker(args.baseline_worker)
    except BaseException:
        label=('smoke_' if args.smoke else '')+(args.worker or args.baseline_worker or ('preflight' if args.preflight else 'resume_queue' if args.resume_queue else 'queue'))
        failure=dict(status='failed',time=now(),pid=os.getpid(),traceback=traceback.format_exc())
        write_json(AUDIT/(label+'_failure.json'),failure)
        p=AUDIT/(label+'_progress.json')
        if p.exists():
            state=json.loads(p.read_text(encoding='utf-8'));state.update(failure);write_json(p,state)
        raise
