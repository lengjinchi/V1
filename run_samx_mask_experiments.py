"""Sequential checks and training: shared Proto/coefficients, then bounded offsets."""
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

from run_samx_experiments import now, write_json
from run_samx_followups import SOURCES as OLD_SOURCES
from samx_mask_experiment_config import (AUDIT, BASELINE_RESULT, MODES, PREVIOUS_AUDIT,
                                         ROOT, options, select_offset_bound, validation_options)

SOURCES = list(dict.fromkeys(OLD_SOURCES + [
    'vajra/nn/samx_mask_variants.py', 'samx_mask_experiment_config.py',
    'run_samx_mask_experiments.py', 'tests/test_samx_mask_variants.py']))


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def check_protected():
    records = json.loads((AUDIT/'protected_weights.json').read_text(encoding='utf-8'))
    for record in records:
        path = ROOT/record['path']
        stat = path.stat()
        assert stat.st_size == record['size'] and stat.st_mtime_ns == record['mtime_ns'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    return dict(checkpoints=len(records), verified_at=now())


def diagnostic():
    """Measure only Proto offsets on 128 fixed training images, without augmentation."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from vajra import Vajra
    from vajra.dataset.utils import check_det_dataset
    from vajra.models.vajra.segment.val import SegmentationValidator
    from vajra.utils.torch_utils import init_seeds
    from diagnose_cfsav1_regions import valid_mask

    destination = AUDIT/'offset_diagnostic.json'
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(4)
    init_seeds(42, deterministic=True)
    previous = json.loads(BASELINE_RESULT.read_text(encoding='utf-8'))
    checkpoint = Path(previous['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == previous['checkpoint_sha256']
    model = Vajra(str(checkpoint)).model.cuda().float().eval()
    assert float(model.model[-1].stride[0]) == 8.
    args = validation_options('offset_statistics')
    args.update(batch=1, cache=False)
    validator = SegmentationValidator(args=args)
    validator.data = check_det_dataset(args['data'])
    validator.stride = 32
    dataset = validator.build_dataset(validator.data['train'], mode='val', batch=1)
    assert len(dataset) == 6134
    indices = np.sort(np.random.default_rng(42).choice(len(dataset), size=128, replace=False)).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=dataset.collate_fn)
    samples, filenames = [], []
    current = {}
    def collect(module, inputs, output):
        assert torch.isfinite(output).all()
        valid = valid_mask(current['batch'], output.shape[-2:]).expand_as(output)
        samples.append(output.detach().float().abs()[valid].cpu().numpy())
    handle = model.model[-1].proto_enhance.geometry.p_conv.register_forward_hook(collect)
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                batch['img'] = batch['img'].cuda().float()/255
                current['batch'] = batch
                model(batch['img'])
                filenames.extend(batch['im_file'])
                if (index+1) % 16 == 0:
                    print(f'OFFSET_DIAGNOSTIC {index+1}/128', flush=True)
    finally:
        handle.remove()
    values = np.concatenate(samples)
    quantiles = np.quantile(values, [.5, .9, .95, .99])
    bound = select_offset_bound(float(quantiles[2]))
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=previous['checkpoint_sha256'],
                  split='train', augmentation=False, sample_seed=42, images=128,
                  sampled_files=filenames, sampled_indices=indices, exclude_padding=True,
                  quantity='absolute offset coordinate before boundary clipping; Proto branch only',
                  units='P3 feature cells', count=int(values.size),
                  quantiles=dict(zip(('p50', 'p90', 'p95', 'p99'), map(float, quantiles))),
                  maximum=float(values.max()), selected_bound=bound,
                  selected_bound_input_pixels=8*bound,
                  fraction_raw_coordinates_exceeding_bound=float((values>bound).mean()),
                  selection_rule='Smallest power of two >= max(1, p95 absolute coordinate), fixed before training; no AP-based selection.',
                  note='r*tanh(o/r) also changes offsets smaller than r; this statistic does not establish that long offsets caused errors.',
                  completed_at=now())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == previous['checkpoint_sha256']
    write_json(destination, result)
    print('OFFSET_DIAGNOSTIC_COMPLETE '+json.dumps(result), flush=True)
    del model, validator, loader, samples, values
    gc.collect(); torch.cuda.empty_cache()


def baseline_validation():
    import torch
    from samx_followup_validation import validate_checkpoint
    torch.set_num_threads(4)
    previous = json.loads(BASELINE_RESULT.read_text(encoding='utf-8'))
    result = validate_checkpoint(previous['checkpoint'], validation_options('baseline_proto_free'),
                                 expected=previous['metrics'])
    result.update(reproduced=True, previous_result=str(BASELINE_RESULT), finished_at=now())
    write_json(AUDIT/'baseline_result.json', result)


def worker(mode, smoke=False):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv
    from vajra.nn.samx import LocalGlobalSAMX
    from vajra.nn.samx_geometry import ResidualScale, GeometricDifference
    from vajra.nn.samx_mask_variants import CoupledProtoSegmentation, BoundedGeometricDifference
    from samx_experiment_config import clear_validation_cache
    from samx_followup_validation import validate_checkpoint

    torch.set_num_threads(4)
    bound = json.loads((AUDIT/'offset_diagnostic.json').read_text(encoding='utf-8'))['selected_bound']
    args = options(mode, smoke, bound)
    prefix = ('smoke_' if smoke else '')+mode
    progress_path = AUDIT/(prefix+'_progress.json')
    run_dir = Path(args['project'])/args['name']
    if progress_path.exists() or run_dir.exists():
        raise FileExistsError(f'Refusing duplicate experiment: {run_dir}')
    manifest = source_hashes()
    write_json(AUDIT/(prefix+'_source_manifest.json'), manifest)
    state = dict(status='initializing', mode=mode, smoke=smoke, pid=os.getpid(),
                 started_at=now(), updated_at=now(), options=args, last_completed_epoch=0)
    write_json(progress_path, state)
    model = Vajra('vajra-v1-nano-seg')
    started = time.monotonic(); last_update = [0.]; batches = [0]; startup = {}
    def verify_start(trainer):
        head = trainer.model.model[-1]
        assert type(trainer.model.model[0].vajra_neck3) is LocalGlobalSAMX
        assert isinstance(head.seg_enhance, nn.Identity)
        assert (type(head) is CoupledProtoSegmentation) == (mode == MODES[0])
        geometry = head.proto_enhance.geometry
        assert type(geometry) is (GeometricDifference if mode == MODES[0] else BoundedGeometricDifference)
        assert head.proto_enhance.mode == 'free' and head.proto_enhance.hidden == 16
        if mode == MODES[1]:
            assert geometry.offset_bound == bound
        for key in ('use_cfsa', 'use_cfsav1', 'cfsav1_global_context', 'cfsav1_semantic_context',
                    'cfsav1_gate_smooth', 'cfsav1_gate_ramp_epochs', 'cfsav1_warmup_epochs'):
            assert not getattr(trainer.args, key), key
        assert not trainer.model.init_criterion().use_morph_mask_loss
        assert not trainer.resume and trainer.start_epoch == 0 and trainer.amp
        assert trainer.epochs == (1 if smoke else 200) and trainer.args.patience == 30
        assert trainer.batch_size == 32 and trainer.args.img_size == 1600 and trainer.args.rect
        assert len(trainer.train_loader.dataset) == (61 if smoke else 6134)
        assert len(trainer.test_loader.dataset) == 1532
        assert Path(trainer.save_dir) == run_dir
        samplers = [m for m in trainer.model.modules() if isinstance(m, LDConv)]
        assert len(samplers) == 3
        for module in samplers:
            assert module.p_conv.weight.count_nonzero() == 0 and module.p_conv.bias.count_nonzero() == 0
        ids = [id(p) for g in trainer.optimizer.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in trainer.model.parameters()}
        offset_ids = {id(p) for m in samplers for p in m.p_conv.parameters()}
        assert offset_ids == {id(p) for g in trainer.optimizer.param_groups if g.get('lr_scale') == .1 for p in g['params']}
        gains = {n:float(m.weight.detach()) for n,m in trainer.model.named_modules() if isinstance(m, ResidualScale)}
        assert len(gains) == 1 and all(abs(v-.1)<1e-6 for v in gains.values())
        startup.update(head_class=type(head).__name__, geometry_class=type(geometry).__name__,
                       share_proto_coeff=args['samx_proto_share_coeff'],
                       offset_bound=args['samx_proto_offset_bound'], parameters=sum(p.numel() for p in trainer.model.parameters()),
                       gains=gains, train_images=len(trainer.train_loader.dataset), val_images=1532,
                       amp=True, optimizer=type(trainer.optimizer).__name__, cfsa=False, cfsav1=False, mabd=False,
                       batch=trainer.batch_size, scratch_verified=True)
        state.update(status='training', startup=startup, updated_at=now())
        write_json(AUDIT/(prefix+'_startup.json'), startup)
        write_json(progress_path, state, best_effort=True)
        print('MASK_STARTUP_VERIFIED '+json.dumps(startup), flush=True)
    def batch_done(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError('Non-finite loss')
        for group in trainer.optimizer.param_groups:
            if group.get('lr_scale') != .1:
                continue
            regular = [g for i,g in enumerate(trainer.optimizer.param_groups) if 'lr_scale' not in g
                       and g['weight_decay'] == group['weight_decay'] and (i==0) == group['warmup_bias']]
            assert regular and abs(group['lr']-.1*regular[0]['lr']) < 1e-10
        batches[0] += 1
        if not startup.get('first_batch_verified'):
            startup.update(first_batch_verified=True, warmup_lr_ratio_verified=True)
            write_json(AUDIT/(prefix+'_startup.json'), startup)
        if time.monotonic()-last_update[0] >= 10:
            last_update[0] = time.monotonic()
            state.update(status='training', updated_at=now(), current_epoch=trainer.epoch+1,
                         batches_seen=batches[0], loss=float(trainer.loss.detach()),
                         elapsed_seconds=time.monotonic()-started,
                         gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(progress_path, state, best_effort=True)
    def val_start(validator):
        clear_validation_cache(validator)
        state.update(status='validating_epoch', updated_at=now())
        write_json(progress_path, state, best_effort=True)
    def epoch_done(trainer):
        state.update(status='training', updated_at=now(), last_completed_epoch=trainer.epoch+1,
                     best_epoch=int(trainer.stopper.best_epoch), best_fitness=float(trainer.best_fitness),
                     metrics={k:float(v) for k,v in (trainer.metrics or {}).items()},
                     elapsed_seconds=time.monotonic()-started,
                     gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(progress_path, state, best_effort=True)
        print('MASK_EPOCH_COMPLETE '+json.dumps(state), flush=True)
    for event, callback in [('on_pretrain_routine_end',verify_start), ('on_train_batch_end',batch_done),
                            ('on_val_start',val_start), ('on_fit_epoch_end',epoch_done)]:
        model.add_callback(event, callback)
    model.train(**args)
    trainer = model.trainer
    state.update(status='validating_fp32_boundary_speed', updated_at=now())
    write_json(progress_path, state, best_effort=True)
    result = validate_checkpoint(trainer.best, validation_options(prefix))
    result.update(mode=mode, smoke=smoke, options=args, startup=startup, finished_at=now(),
                  completed_epochs=trainer.epoch+1, best_epoch=trainer.stopper.best_epoch,
                  elapsed_seconds=time.monotonic()-started, source_unchanged=source_hashes()==manifest,
                  training_peak_allocated_gib=state['gpu_peak_allocated_gib'])
    write_json(AUDIT/(prefix+'_result.json'), result)
    assert result['source_unchanged'], 'Sources changed during experiment; result retained'
    check_protected()
    state.update(status='complete', updated_at=now(), result_file=str(AUDIT/(prefix+'_result.json')))
    write_json(progress_path, state)
    print('MASK_EXPERIMENT_COMPLETE '+json.dumps(result), flush=True)


def update_summary():
    baseline = json.loads(BASELINE_RESULT.read_text(encoding='utf-8'))
    records = dict(proto_free=baseline)
    for mode in MODES:
        path = AUDIT/(mode+'_result.json')
        if path.exists():
            records[mode] = json.loads(path.read_text(encoding='utf-8'))
    keys = ['metrics/precision(Mask)', 'metrics/recall(Mask)', 'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)']
    deltas = {mode:{key:100*(r['metrics'][key]-baseline['metrics'][key]) for key in keys}
              for mode,r in records.items() if mode != 'proto_free'}
    write_json(AUDIT/'comparison.json', dict(records=records,deltas_percentage_points=deltas,
        note='Independent scratch trials against proto_free; best checkpoint selected by unchanged box+mask fitness. Single seed.'))
    lines = ['# SAMX 原型/系数协同与有界偏移实验', '',
             '两项独立从头训练，均对比已完成的 proto_free；旧模块与权重保留。未完成的实验不填入结果表。', '',
             '| 方法 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 | 纯前向 FPS |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode, result in records.items():
        values = [f"{100*result['metrics'][key]:.2f}%" for key in keys]
        lines.append('| '+mode+' | '+' | '.join(values)+f" | {100*result['boundary']['overall']['f1']:.2f}% | {result['forward_benchmark']['fps']:.2f} |")
    lines += ['', '评价沿用前轮完整 FP32 验证、含未匹配实例的边界 F1，以及相同形状的纯前向测速。',
              '重点看 Mask Recall/AP50 能否恢复，同时保住 AP50:95、边界 F1，并报告速度代价。小幅单种子差异不作显著性结论。']
    (AUDIT/'结果汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def child(arguments, label, state):
    logpath = AUDIT/(label+'.log')
    with logpath.open('x', encoding='utf-8') as logfile:
        process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*arguments],
            cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'},
            stdout=logfile,stderr=subprocess.STDOUT)
        state.update(worker_pid=process.pid, current_task=label, updated_at=now())
        state.pop('worker_progress', None)
        write_json(AUDIT/'status.json', state)
        while process.poll() is None:
            progress = AUDIT/(label+'_progress.json')
            if progress.exists():
                try:
                    state['worker_progress'] = json.loads(progress.read_text(encoding='utf-8'))
                except (OSError,json.JSONDecodeError):
                    pass
            state['updated_at'] = now()
            write_json(AUDIT/'status.json', state, best_effort=True)
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{label} exited {process.returncode}; inspect {logpath}')


def queue():
    if (AUDIT/'status.json').exists():
        raise FileExistsError('Queue already exists; refusing duplicate training')
    previous = json.loads((PREVIOUS_AUDIT/'status.json').read_text(encoding='utf-8'))
    assert previous['status'] == 'complete'
    tests = json.loads((AUDIT/'tests_result.json').read_text(encoding='utf-8'))
    extra = json.loads((AUDIT/'new_tests_result.json').read_text(encoding='utf-8'))
    assert tests['success'] and extra['success']
    manifest = source_hashes()
    assert extra['source_manifest'] == manifest, 'Sources changed since targeted tests'
    write_json(AUDIT/'source_manifest.json', manifest)
    with zipfile.ZipFile(AUDIT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT/name,name)
    state = dict(status='running', phase='preflight', pid=os.getpid(), started_at=now(),
                 updated_at=now(), modes=list(MODES), completed=[], preflight_completed=[])
    try:
        check_protected()
        steps = [(['--diagnostic'],'offset_diagnostic'), (['--baseline-validation'],'baseline_validation')]
        steps += [(['--worker',mode,'--smoke'],'smoke_'+mode) for mode in MODES]
        for arguments, label in steps:
            assert source_hashes() == manifest, 'Sources changed before preflight child'
            child(arguments, label, state)
            state['preflight_completed'].append(label)
        assert source_hashes() == manifest
        write_json(AUDIT/'preflight_complete.json',dict(finished_at=now(),steps=state['preflight_completed'],
            manifest=manifest,protected=check_protected()))
        state['phase'] = 'training'
        update_summary()
        for mode in MODES:
            assert source_hashes() == manifest
            check_protected()
            child(['--worker',mode], mode, state)
            result = json.loads((AUDIT/(mode+'_result.json')).read_text(encoding='utf-8'))
            assert result['source_unchanged']
            state['completed'].append(mode)
            update_summary()
        state.update(status='complete', updated_at=now(), current_task=None, protected=check_protected())
        write_json(AUDIT/'status.json', state)
    except BaseException:
        state.update(status='failed', updated_at=now(), traceback=traceback.format_exc())
        write_json(AUDIT/'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--queue', action='store_true')
    group.add_argument('--diagnostic', action='store_true')
    group.add_argument('--baseline-validation', action='store_true')
    group.add_argument('--worker', choices=MODES)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and not args.worker:
        parser.error('--smoke requires --worker')
    try:
        if args.queue:
            queue()
        elif args.diagnostic:
            diagnostic()
        elif args.baseline_validation:
            baseline_validation()
        else:
            worker(args.worker, args.smoke)
    except BaseException:
        label = (('smoke_' if args.smoke else '')+args.worker if args.worker else
                 'queue' if args.queue else 'offset_diagnostic' if args.diagnostic else 'baseline_validation')
        failure = dict(status='failed', updated_at=now(), pid=os.getpid(), traceback=traceback.format_exc())
        write_json(AUDIT/(label+'_failure.json'), failure)
        if args.worker:
            progress = AUDIT/(('smoke_' if args.smoke else '')+args.worker+'_progress.json')
            if progress.exists():
                state = json.loads(progress.read_text(encoding='utf-8'))
                state.update(failure)
                write_json(progress,state,best_effort=True)
        raise
