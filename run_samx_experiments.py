"""Run experiment 2 then 3 in isolated processes, with durable logs/status.

Original repaired SAMX is deliberately excluded from this queue.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone

from samx_experiment_config import AUDIT, ROOT, EXPERIMENTS, training_options

MODES = ("stage_global", "local_global")
SOURCES = ["vajra/nn/modules.py", "vajra/nn/samx.py", "vajra/nn/vajra.py",
           "vajra/core/trainer.py", "vajra/core/model.py", "vajra/loss/__init__.py",
           "vajra/models/vajra/segment/train.py", "vajra/configs/__init__.py",
           "vajra/configs/hyperparams/default_hyp.yaml", "samx_experiment_config.py",
           "run_samx_experiments.py"]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value, *, best_effort=False):
    """Atomically replace JSON, tolerating transient Windows reader locks.

    Only expendable progress updates may opt out of raising after retries.
    Checkpoint metadata and final results remain strict by default.
    """
    path = Path(path)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        for attempt in range(8):
            try:
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.02 * 2 ** attempt, .2))
    except OSError as exc:
        if not best_effort:
            raise
        warnings.warn(f"Progress update skipped; training continues: {path}: {exc}", RuntimeWarning)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def run_worker(mode, smoke=False):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv
    from vajra.nn.samx import LocalGlobalSAMX, StageGlobalSAMX

    torch.set_num_threads(4)
    options = training_options(mode, smoke=smoke)
    prefix = ("smoke_" if smoke else "") + mode
    progress_path = AUDIT / (prefix + "_progress.json")
    record = dict(mode=mode, smoke=smoke, status="initializing", updated_at=now(),
                  options=options, pid=os.getpid(), started_at=now())
    write_json(progress_path, record)
    run_dir = Path(options['project'])/options['name']
    if run_dir.exists() and not smoke:
        raise FileExistsError(f"Refusing to overwrite or silently rename experiment: {run_dir}")
    model = Vajra("vajra-v1-nano-seg")
    captured = {}

    def verify_start(trainer):
        neck = trainer.model.model[0].vajra_neck3
        expected = StageGlobalSAMX if mode == 'stage_global' else LocalGlobalSAMX
        assert type(neck) is expected, (type(neck), expected)
        head = trainer.model.model[-1]
        assert isinstance(head.proto_enhance, nn.Identity)
        assert isinstance(head.seg_enhance, nn.Identity)
        assert not head.use_cfsav1 and not head.use_fft_guided_ema and not head.use_fft_guided_ema_proto_only
        assert not trainer.args.use_cfsa and not trainer.args.use_cfsav1
        assert not trainer.model.init_criterion().use_morph_mask_loss
        assert trainer.amp, "AMP unexpectedly disabled; stop rather than change experiment settings"
        offsets = {id(p) for m in trainer.model.modules() if isinstance(m, LDConv) for p in m.p_conv.parameters()}
        assert len(offsets) == 4
        ids = [id(p) for g in trainer.optimizer.param_groups for p in g['params']]
        assert len(ids) == len(set(ids))
        slow = [g for g in trainer.optimizer.param_groups if g.get('lr_scale') == .1]
        assert {id(p) for g in slow for p in g['params']} == offsets
        captured.update(neck_class=type(neck).__name__, offset_parameter_tensors=len(offsets),
                        parameters=sum(p.numel() for p in trainer.model.parameters()),
                        cfsa_enabled=False, cfsav1_enabled=False, mabd_enabled=False,
                        amp=trainer.amp, run_dir=str(trainer.save_dir),
                        optimizer=type(trainer.optimizer).__name__,
                        optimizer_groups=[dict(lr=g['lr'], decay=g['weight_decay'],
                            lr_scale=g.get('lr_scale',1), warmup_bias=g.get('warmup_bias',i==0),
                            tensors=len(g['params'])) for i,g in enumerate(trainer.optimizer.param_groups)])
        record.update(status="training", startup=captured, updated_at=now())
        write_json(AUDIT/(prefix+"_startup.json"), captured)
        write_json(progress_path, record)

    def verify_batch(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Non-finite SAMX loss")
        for g in trainer.optimizer.param_groups:
            if g.get('lr_scale') != .1:
                continue
            matching = [b for i,b in enumerate(trainer.optimizer.param_groups)
                        if 'lr_scale' not in b and b['weight_decay'] == g['weight_decay']
                        and (i==0) == g['warmup_bias']]
            assert matching, "Missing corresponding main optimizer group"
            assert abs(g['lr'] - .1*matching[0]['lr']) < 1e-10
        captured['warmup_ratio_verified'] = True
        if not captured.get('first_batch_verified'):
            captured['first_batch_verified'] = True
            captured['batch_size'] = trainer.batch_size
            captured['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            write_json(AUDIT/(prefix+"_startup.json"), captured)

    def epoch_done(trainer):
        record.update(status="training", updated_at=now(), last_completed_epoch=trainer.epoch+1,
                      best_fitness=float(trainer.best_fitness) if trainer.best_fitness is not None else None,
                      metrics={k:float(v) for k,v in (trainer.metrics or {}).items()},
                      gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(progress_path, record)

    model.add_callback('on_pretrain_routine_end', verify_start)
    model.add_callback('on_train_batch_end', verify_batch)
    model.add_callback('on_fit_epoch_end', epoch_done)
    model.train(**options)
    trainer = model.trainer
    checkpoint = trainer.best
    assert checkpoint.exists(), f"No best checkpoint at {checkpoint}"
    record.update(status="validating_fp32", updated_at=now(), checkpoint=str(checkpoint))
    write_json(progress_path, record)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validation_options = dict(data=options['data'], split="val", img_size=1600, rect=True,
        batch=8, workers=0, device=0, half=False, conf=.001, iou=.7, max_det=300,
        overlap_mask=True, mask_ratio=4, augment=False, plots=False, verbose=False,
        project=str(AUDIT/"validation"), name=prefix, exist_ok=False)
    validated = Vajra(str(checkpoint))
    validation_counts = {}
    def count_validation(validator):
        validation_counts.update(images=int(validator.seen),
                                 instances=int(validator.nt_per_class.sum()))
    validated.add_callback('on_val_end', count_validation)
    metrics = validated.val(**validation_options)
    assert validation_counts == dict(images=1532, instances=3942), validation_counts
    assert checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = dict(mode=mode, smoke=smoke, finished_at=now(), checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, options=options,
                  validation_options=validation_options, startup=captured,
                  validation_counts=validation_counts, completed_epochs=int(trainer.epoch)+1,
                  best_epoch=int(trainer.stopper.best_epoch),
                  metrics={k:float(v) for k,v in metrics.results_dict.items()},
                  speed={k:float(v) for k,v in metrics.speed.items()})
    write_json(AUDIT/(prefix+"_result.json"), result)
    record.update(status="complete", updated_at=now(), result_file=str(AUDIT/(prefix+"_result.json")))
    write_json(progress_path, record)
    print(json.dumps(result, indent=2), flush=True)


def create_comparison():
    results=[json.loads((AUDIT/(m+"_result.json")).read_text(encoding='utf-8')) for m in MODES]
    keys=['metrics/precision(Mask)','metrics/recall(Mask)','metrics/mAP50(Mask)','metrics/mAP50-95(Mask)']
    rows=[]
    for r in results:
        rows.append(dict(mode=r['mode'], **{k: r['metrics'][k]*100 for k in keys}))
    write_json(AUDIT/'comparison.json', dict(results=results, percentage_values=rows,
        note="Only experiments 2 and 3 were trained. No repaired-original control; single seed."))
    with (AUDIT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['mode']+keys);writer.writeheader();writer.writerows(rows)
    lines=['# SAMX 两项实验结果','', '两项均关闭 CFSA/CFSAv1，应用相同 LDConv 修复。以下为最佳权重的统一独立 FP32 验证。', '',
        '| 实验 | Mask P | Mask R | Mask AP50 | Mask AP50:95 |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| '+row['mode']+' | '+' | '.join(f'{row[k]:.4f}%' for k in keys)+' |')
    lines += ['', '本次未训练修复后的原版对照；不能将相对未修复历史模型的变化全部归因于门控。单种子结果不代表统计显著。']
    (AUDIT/'结果对比.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_queue():
    AUDIT.mkdir(parents=True,exist_ok=True)
    manifest_path=AUDIT/'source_manifest.json'
    if (AUDIT/'status.json').exists():
        raise FileExistsError('Queue status already exists; inspect it instead of starting duplicate training')
    manifest=source_hashes();write_json(manifest_path,manifest)
    state=dict(status='running', pid=os.getpid(), started_at=now(), modes=list(MODES),
               original_control_queued=False, completed=[])
    try:
        for mode in MODES:
            if source_hashes()!=manifest:
                raise RuntimeError('Experiment source changed after queue start; stop for reproducibility')
            state.update(current_mode=mode, updated_at=now())
            write_json(AUDIT/'status.json',state)
            env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
            with (AUDIT/(mode+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',mode],
                    cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
                state['worker_pid']=child.pid
                while child.poll() is None:
                    progress=AUDIT/(mode+'_progress.json')
                    if progress.exists():
                        try: state['progress']=json.loads(progress.read_text(encoding='utf-8'))
                        except (OSError,json.JSONDecodeError): pass
                    state['updated_at']=now();write_json(AUDIT/'status.json',state)
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{mode} failed with exit code {child.returncode}; inspect {mode}.log')
            state['completed'].append(mode)
        create_comparison()
        state.update(status='complete', current_mode=None, updated_at=now())
        write_json(AUDIT/'status.json',state)
    except BaseException as exc:
        state.update(status='failed', error=str(exc), updated_at=now())
        write_json(AUDIT/'status.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',action='store_true')
    parser.add_argument('--worker',choices=MODES)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.queue:
        if args.worker or args.smoke:parser.error('--queue is separate from --worker/--smoke')
        run_queue()
    elif args.worker:
        try: run_worker(args.worker,args.smoke)
        except BaseException:
            write_json(AUDIT/(("smoke_" if args.smoke else "")+args.worker+'_failure.json'),
                       dict(time=now(), traceback=traceback.format_exc()))
            raise
    else:parser.error('Choose --queue or --worker MODE')
