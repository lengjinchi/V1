"""Run a real-data smoke check, then the PPT joint model, in separate processes.

Usage: python -u run_joint_experiment.py --queue
Progress, logs, code snapshot and results are written to the dedicated audit folder.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from joint_experiment_config import (
    AUDIT, NAME, PRESENTATION, ROOT, clear_validation_cache,
    training_options, validation_options,
)
from run_samx_experiments import now, write_json


def sources():
    return sorted(set(
        [p for p in (ROOT / 'vajra').rglob('*')
         if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]
        + [ROOT / name for name in (
            'joint_experiment_config.py', 'run_joint_experiment.py',
            'samx_experiment_config.py', 'run_samx_experiments.py')]
    ))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources()}


def verify_structure(model, *, check_training_loss=True):
    import torch
    from torch import nn
    from vajra.nn.cfsav1 import CFSAv1
    from vajra.nn.modules import LDConv
    from vajra.nn.samx import LocalGlobalSAMX

    neck = model.model[0].vajra_neck3
    head = model.model[-1]
    cfsa = head.proto_enhance
    assert type(neck) is LocalGlobalSAMX, type(neck)
    assert [name for name, m in model.named_modules() if isinstance(m, LocalGlobalSAMX)] == ['model.0.vajra_neck3']
    assert type(cfsa) is CFSAv1, type(cfsa)
    assert sum(isinstance(m, CFSAv1) for m in model.modules()) == 1
    assert cfsa.rho == 0.25 and cfsa.groups == 4 and cfsa.context_channels == 16
    assert not cfsa.global_context and cfsa.global_gate is None
    assert not cfsa.semantic_channels and cfsa.semantic_gate is None
    assert not cfsa.gate_smooth and float(cfsa.gate_scale) == 1.0
    assert not getattr(cfsa, 'use_sensitivity', False)
    assert head.use_cfsav1 and not head.use_fft_guided_ema and not head.use_fft_guided_ema_proto_only
    assert isinstance(head.seg_enhance, nn.Identity)
    # Loaded inference models store args as a dict; criterion construction is
    # a trainer operation and expects its namespace. Check the loss at startup.
    if check_training_loss:
        assert not model.init_criterion().use_morph_mask_loss
    offsets = {id(p) for m in model.modules() if isinstance(m, LDConv) for p in m.p_conv.parameters()}
    assert len(offsets) == 4
    return neck, cfsa, offsets


def run_worker(smoke):
    import torch
    from vajra import Vajra

    torch.set_num_threads(4)
    phase = 'smoke' if smoke else 'train'
    options = training_options(smoke)
    destination = Path(options['project']) / options['name']
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite or silently rename: {destination}')
    progress_path = AUDIT / f'{phase}_progress.json'
    record = dict(status='initializing', phase=phase, pid=os.getpid(),
                  started_at=now(), updated_at=now(), options=options)
    write_json(progress_path, record)
    startup, snapshots, monitored = {}, {}, {}
    calls = {'samx': 0, 'cfsav1': 0}
    handles = []
    model = Vajra('vajra-v1-nano-seg')

    def check_start(trainer):
        neck, cfsa, offsets = verify_structure(trainer.model)
        assert trainer.amp, 'AMP disabled unexpectedly'
        assert trainer.args.samx_mode == 'local_global'
        assert not trainer.args.use_cfsa and trainer.args.use_cfsav1
        assert trainer.args.samx_proto_mode == 'off'
        assert trainer.args.cfsav1_gate_ramp_epochs == trainer.args.cfsav1_warmup_epochs == 0
        if not smoke:
            assert len(trainer.train_loader.dataset) == 6134
        ids = [id(p) for group in trainer.optimizer.param_groups for p in group['params']]
        assert len(ids) == len(set(ids)), 'Duplicate optimizer parameter'
        learnable = {id(p) for m in (neck, cfsa) for p in m.parameters() if p.requires_grad}
        assert learnable.issubset(set(ids)), 'Joint parameters missing from optimizer'
        slow = [g for g in trainer.optimizer.param_groups if g.get('lr_scale') == 0.1]
        assert {id(p) for g in slow for p in g['params']} == offsets
        monitored.update(cfsav1_gate=cfsa.gate[-1].weight,
                         samx_local_gate=neck.local_route[-1].weight)
        assert all(p.requires_grad for p in monitored.values())
        snapshots.update({k: p.detach().cpu().clone() for k, p in monitored.items()})
        def count(which):
            def hook(module, inputs, output):
                if module.training:
                    calls[which] += 1
            return hook
        handles.extend([neck.register_forward_hook(count('samx')),
                        cfsa.register_forward_hook(count('cfsav1'))])
        startup.update(neck_class=type(neck).__name__, cfsav1_class=type(cfsa).__name__,
            cfsav1_global=False, cfsav1_semantic=False, cfsav1_rho=cfsa.rho,
            cfsav1_smooth=False, cfsav1_gate_scale=float(cfsa.gate_scale),
            cfsa_enabled=False, mabd_enabled=False, offset_parameter_tensors=len(offsets),
            amp=bool(trainer.amp), optimizer=type(trainer.optimizer).__name__,
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            train_images=len(trainer.train_loader.dataset),
            train_batches=len(trainer.train_loader), batch_size=trainer.batch_size,
            run_dir=str(trainer.save_dir))
        record.update(status='training', startup=startup, updated_at=now())
        write_json(AUDIT / f'{phase}_startup.json', startup)
        write_json(progress_path, record)
        print('JOINT_STARTUP ' + json.dumps(startup), flush=True)

    batches = 0
    def check_batch(trainer):
        nonlocal batches
        batches += 1
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError('Non-finite joint loss')
        if smoke or batches <= 20:
            print('JOINT_STEP ' + json.dumps(dict(batch=batches,
                amp_scale=trainer.scaler.get_scale(),
                gate_updates={key: float((p.detach().cpu() - snapshots[key]).abs().max())
                              for key, p in monitored.items()})), flush=True)
        for group in trainer.optimizer.param_groups:
            if group.get('lr_scale') != 0.1:
                continue
            matching = [g for i, g in enumerate(trainer.optimizer.param_groups)
                        if 'lr_scale' not in g and g['weight_decay'] == group['weight_decay']
                        and (i == 0) == group['warmup_bias']]
            assert matching and abs(group['lr'] - 0.1 * matching[0]['lr']) < 1e-10
        if batches == 1 or batches % 10 == 0:
            record.update(status='training', updated_at=now(), epoch=trainer.epoch + 1,
                total_batches_processed=batches, finite_loss=float(trainer.loss.detach()),
                amp_scale=trainer.scaler.get_scale(),
                train_forward_calls=calls.copy(),
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            write_json(progress_path, record, best_effort=True)

    def check_epoch(trainer):
        if trainer.epoch != 0:
            return
        changes = {key: float((p.detach().cpu() - snapshots[key]).abs().max())
                   for key, p in monitored.items()}
        assert calls['samx'] > 0 and calls['cfsav1'] > 0, calls
        assert all(value > 0 and value < float('inf') for value in changes.values()), changes
        # Full-model checkpoints pickle hooks. These temporary local callbacks
        # must be removed before the first validation/checkpoint save.
        for handle in handles:
            handle.remove()
        handles.clear()
        startup.update(first_epoch_parameter_updates=changes,
                       train_forward_calls=calls.copy(), offset_lr_ratio_verified=True)
        write_json(AUDIT / f'{phase}_startup.json', startup)
        print('JOINT_BACKWARD_VERIFIED ' + json.dumps(changes), flush=True)

    def epoch_done(trainer):
        record.update(status='training', updated_at=now(), last_completed_epoch=trainer.epoch + 1,
            best_fitness=float(trainer.best_fitness) if trainer.best_fitness is not None else None,
            metrics={k: float(v) for k, v in (trainer.metrics or {}).items()},
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(progress_path, record, best_effort=True)

    model.add_callback('on_pretrain_routine_end', check_start)
    model.add_callback('on_train_batch_end', check_batch)
    model.add_callback('on_train_epoch_end', check_epoch)
    model.add_callback('on_fit_epoch_end', epoch_done)
    def val_start(validator):
        clear_validation_cache(validator)
        record.update(status='validating_epoch', updated_at=now())
        write_json(progress_path, record, best_effort=True)
    model.add_callback('on_val_start', val_start)
    model.train(**options)
    for handle in handles:
        handle.remove()
    checkpoint = Path(model.trainer.best)
    assert checkpoint.exists(), checkpoint
    completed_epochs = int(model.trainer.epoch) + 1
    best_epoch = int(model.trainer.stopper.best_epoch)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record.update(status='validating_fp32', checkpoint=str(checkpoint), updated_at=now())
    write_json(progress_path, record)
    # Release training optimizer/EMA before the separate full FP32 validation.
    del model
    monitored.clear()
    gc.collect()
    torch.cuda.empty_cache()
    validated = Vajra(str(checkpoint))
    verify_structure(validated.model, check_training_loss=False)
    counts = {}
    def count_validation(validator):
        counts.update(images=int(validator.seen), instances=int(validator.nt_per_class.sum()))
    validated.add_callback('on_val_start', clear_validation_cache)
    validated.add_callback('on_val_end', count_validation)
    val_options = validation_options(phase)
    metrics = validated.val(**val_options)
    assert counts == dict(images=1532, instances=3942), counts
    assert checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = dict(phase=phase, smoke=smoke, finished_at=now(), checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash, options=options, validation_options=val_options,
        validation_counts=counts, startup=startup, completed_epochs=completed_epochs,
        best_epoch=best_epoch, metrics={k: float(v) for k, v in metrics.results_dict.items()},
        speed={k: float(v) for k, v in metrics.speed.items()})
    write_json(AUDIT / f'{phase}_result.json', result)
    record.update(status='complete', updated_at=now(), result_file=str(AUDIT / f'{phase}_result.json'))
    write_json(progress_path, record)
    print('JOINT_RESULT ' + json.dumps(result), flush=True)


def run_queue():
    AUDIT.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / 'status.json'
    if status_path.exists():
        raise FileExistsError(f'Existing queue; inspect status before starting another: {status_path}')
    state = dict(status='initializing', pid=os.getpid(), started_at=now(), completed=[])
    write_json(status_path, state)
    try:
        manifest = source_hashes()
        data = Path(training_options()['data'])
        write_json(AUDIT / 'source_manifest.json', manifest)
        write_json(AUDIT / 'experiment.json', dict(
            presentation=str(PRESENTATION),
            presentation_sha256=hashlib.sha256(PRESENTATION.read_bytes()).hexdigest(),
            data_yaml=str(data), data_yaml_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            training_options=training_options(), smoke_options=training_options(True),
            validation_options=validation_options('train'),
            note='CFSAv1 local-only at Proto + LocalGlobalSAMX at P4. No old runs are overwritten.'))
        with zipfile.ZipFile(AUDIT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sources():
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.write(PRESENTATION, 'reference/' + PRESENTATION.name)
            archive.write(data, 'reference/data.yaml')
        for phase in ('smoke', 'train'):
            assert source_hashes() == manifest, 'Source changed after experiment snapshot'
            options = training_options(phase == 'smoke')
            destination = Path(options['project']) / options['name']
            if destination.exists():
                raise FileExistsError(destination)
            state.update(status='running', phase=phase, updated_at=now())
            write_json(status_path, state)
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
            with (AUDIT / f'{phase}.log').open('x', encoding='utf-8') as log:
                child = subprocess.Popen(
                    [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', phase],
                    cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                state['worker_pid'] = child.pid
                while child.poll() is None:
                    progress = AUDIT / f'{phase}_progress.json'
                    if progress.exists():
                        try:
                            state['progress'] = json.loads(progress.read_text(encoding='utf-8'))
                        except (OSError, json.JSONDecodeError):
                            pass
                    state['updated_at'] = now()
                    write_json(status_path, state, best_effort=True)
                    time.sleep(3)
                if child.returncode:
                    raise RuntimeError(f'{phase} failed (exit {child.returncode}); inspect {phase}.log')
            assert source_hashes() == manifest, 'Source changed while experiment was running'
            assert (AUDIT / f'{phase}_result.json').exists()
            state['completed'].append(phase)
        state.update(status='complete', phase=None, updated_at=now(),
                     result_file=str(AUDIT / 'train_result.json'))
        write_json(status_path, state)
    except BaseException as exc:
        state.update(status='failed', error=str(exc), updated_at=now())
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--queue', action='store_true')
    group.add_argument('--worker', choices=('smoke', 'train'))
    args = parser.parse_args()
    if args.queue:
        run_queue()
    else:
        try:
            run_worker(args.worker == 'smoke')
        except BaseException:
            failure = dict(status='failed', time=now(), traceback=traceback.format_exc())
            write_json(AUDIT / f'{args.worker}_failure.json', failure)
            progress = AUDIT / f'{args.worker}_progress.json'
            if progress.exists():
                state = json.loads(progress.read_text(encoding='utf-8'))
                state.update(failure, updated_at=now())
                write_json(progress, state)
            raise
