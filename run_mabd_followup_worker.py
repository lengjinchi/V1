"""One isolated smoke/train worker for MABD follow-ups, repeats or joint pairs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

from mabd_followup_config import AUDIT, training_options, validation_options
from run_samx_experiments import now, write_json


def run_job(job, smoke):
    import torch
    from torch import nn
    from vajra import Vajra
    from run_mabd_experiment import evaluate_checkpoint
    from samx_experiment_config import clear_validation_cache
    torch.set_num_threads(4)
    phase='smoke' if smoke else 'train'
    tag=job['tag']
    job_dir=AUDIT/'jobs'/tag
    job_dir.mkdir(parents=True,exist_ok=True)
    options=training_options(tag,seed=job.get('seed',42),smoke=smoke,gain=job.get('gain',.15),
                             joint=job.get('joint',False),original=job.get('original',False))
    destination=Path(options['project'])/options['name']
    if destination.exists():
        raise FileExistsError(destination)
    path=job_dir/f'{phase}_progress.json'
    record=dict(status='initializing',pid=os.getpid(),started_at=now(),updated_at=now(),options=options)
    write_json(path,record)
    model=Vajra('vajra-v1-nano-seg')
    monitored,snapshots,startup={},{},{}
    batches=0

    def check_start(trainer):
        head=trainer.model.model[-1]
        criterion=trainer.model.init_criterion()
        assert criterion.mask_loss_mode==options['mask_loss_mode']
        assert criterion.use_morph_mask_loss and criterion.seg_gain==7.5
        trainer.model.criterion=criterion
        assert trainer.amp and type(trainer.optimizer).__name__=='SGD'
        assert len(trainer.train_loader.dataset)==(613 if smoke else 6134)
        assert len(trainer.test_loader.dataset)==1532
        assert trainer.save_dir.resolve()==destination.resolve()
        assert trainer.args.samx_proto_mode=='off' and not trainer.args.use_cfsa
        monitored['proto_output']=head.proto.conv3.conv.weight
        if job.get('joint'):
            from run_joint_experiment import verify_structure
            neck,cfsa,offsets=verify_structure(trainer.model,check_training_loss=False)
            monitored.update(cfsav1_gate=cfsa.gate[-1].weight,samx_local_gate=neck.local_route[-1].weight)
            slow={id(p) for group in trainer.optimizer.param_groups if group.get('lr_scale')==.1 for p in group['params']}
            assert slow==offsets
        else:
            assert type(trainer.model.model[0].vajra_neck3).__name__=='VajraV1MerudandaX'
            assert trainer.args.samx_mode=='off' and not trainer.args.use_cfsav1
            assert isinstance(head.proto_enhance,nn.Identity) and isinstance(head.seg_enhance,nn.Identity)
        optimizer_ids=[id(p) for group in trainer.optimizer.param_groups for p in group['params']]
        assert len(optimizer_ids)==len(set(optimizer_ids))
        assert {id(p) for p in trainer.model.parameters() if p.requires_grad}.issubset(set(optimizer_ids))
        snapshots.update({key:p.detach().cpu().clone() for key,p in monitored.items()})
        startup.update(mask_loss_mode=criterion.mask_loss_mode,dice_weight_source='original_bbox',
            boundary_base=options['mabd_v1_boundary_base'],boundary_gain=options['mabd_v1_boundary_gain'],
            boundary_cap=options['mabd_v1_boundary_cap'],expanded_roi=False,
            joint=bool(job.get('joint')),original=bool(job.get('original')),scratch=True,
            seed=options['seed'],train_images=len(trainer.train_loader.dataset),val_images=1532,
            train_batches=len(trainer.train_loader),amp=bool(trainer.amp),optimizer='SGD',
            parameters=sum(p.numel() for p in trainer.model.parameters()),run_dir=str(destination))
        write_json(job_dir/f'{phase}_startup.json',startup)
        record.update(status='training',startup=startup,updated_at=now());write_json(path,record)
        print('FOLLOWUP_STARTUP '+json.dumps(startup),flush=True)

    def batch_done(trainer):
        nonlocal batches
        batches+=1
        assert torch.isfinite(trainer.loss).all() and torch.isfinite(trainer.loss_items).all()
        criterion=trainer.model.criterion
        stats=getattr(criterion,'mabd_v1_last_diagnostics',None)
        if not job.get('original'):
            assert stats is not None and all(torch.isfinite(v).all() for v in stats.values())
            assert float(stats['outer_boundary_pixels'])==0
            assert abs(float(stats['dice_weight_mean'])-float(stats['original_dice_weight_mean']))<1e-6
            assert 0<=float(stats['boundary_weight_mean'])<=.150001
            if job.get('gain',.15)==.15:
                assert abs(float(stats['boundary_weight_mean'])-.15*float(stats['thinness_mean']))<1e-6
        if batches==1 or batches%10==0:
            record.update(status='training',updated_at=now(),epoch=trainer.epoch+1,
                batches=batches,finite_loss=float(trainer.loss.detach()),
                loss_items=[float(v) for v in trainer.loss_items],amp_scale=trainer.scaler.get_scale(),
                actual_loss_calls=getattr(criterion,'mabd_v1_calls',None),
                diagnostics={k:float(v) for k,v in (stats or {}).items()},
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(path,record,best_effort=True)

    def epoch_done(trainer):
        if trainer.epoch==0:
            updates={key:float((p.detach().cpu()-snapshots[key]).abs().max()) for key,p in monitored.items()}
            assert all(0<v<float('inf') for v in updates.values()),updates
            startup['first_epoch_parameter_updates']=updates
            write_json(job_dir/f'{phase}_startup.json',startup)
            print('FOLLOWUP_BACKWARD_VERIFIED '+json.dumps(updates),flush=True)

    def fit_done(trainer):
        record.update(status='training',updated_at=now(),last_completed_epoch=trainer.epoch+1,
            best_epoch=int(trainer.stopper.best_epoch),
            metrics={k:float(v) for k,v in (trainer.metrics or {}).items()})
        write_json(path,record,best_effort=True)

    model.add_callback('on_pretrain_routine_end',check_start)
    model.add_callback('on_train_batch_end',batch_done)
    model.add_callback('on_train_epoch_end',epoch_done)
    model.add_callback('on_fit_epoch_end',fit_done)
    model.add_callback('on_val_start',clear_validation_cache)
    model.train(**options)
    checkpoint=Path(model.trainer.best)
    epochs=int(model.trainer.epoch)+1;best_epoch=int(model.trainer.stopper.best_epoch)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key in ('mask_loss_mode','mabd_v1_boundary_base','mabd_v1_boundary_gain','mabd_v1_boundary_cap','seed'):
        assert saved['train_args'][key]==options[key]
    del saved,model
    monitored.clear();gc.collect();torch.cuda.empty_cache()
    record.update(status='validating_fp32_boundary',updated_at=now());write_json(path,record)
    result=evaluate_checkpoint(checkpoint,f'{tag}_{phase}',options=validation_options(f'{tag}_{phase}'))
    result.update(job=job,options=options,smoke=smoke,completed_epochs=epochs,best_epoch=best_epoch,startup=startup)
    write_json(job_dir/f'{phase}_result.json',result)
    record.update(status='complete',updated_at=now(),result_file=str(job_dir/f'{phase}_result.json'))
    write_json(path,record)
    print('FOLLOWUP_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    job=json.loads(Path(args.job).read_text(encoding='utf-8'))
    try:
        run_job(job,args.smoke)
    except BaseException:
        write_json(AUDIT/'jobs'/job['tag']/('smoke_failure.json' if args.smoke else 'train_failure.json'),
                   dict(status='failed',time=now(),traceback=traceback.format_exc()))
        raise
