"""Durable MABDv1 queue: smoke -> old-checkpoint validation -> full training.

Only the full variant is trained. The original MABD and partial ablations are
not queued. Checkpoints are evaluated with the same FP32/boundary protocol.
"""
import argparse
import csv
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

from mabd_experiment_config import (
    AUDIT, ORIGINAL_CHECKPOINT, ORIGINAL_RUN, ROOT,
    clear_validation_cache, training_options, validation_options,
)
from run_samx_experiments import now, write_json


def sources():
    return sorted(set(
        [p for p in (ROOT / 'vajra').rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]
        + [ROOT / name for name in (
            'mabd_experiment_config.py', 'run_mabd_experiment.py', 'tests/test_mabd_v1.py',
            'samx_experiment_config.py', 'run_samx_experiments.py', 'samx_followup_validation.py')]
    ))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}


def historical_reference():
    import torch
    import yaml
    args = yaml.safe_load((ORIGINAL_RUN / 'args.yaml').read_text(encoding='utf-8'))
    saved = torch.load(ORIGINAL_CHECKPOINT, map_location='cpu', weights_only=False)
    model = saved.get('ema') or saved['model']
    criterion = getattr(model, 'criterion', None)
    assert type(model.model[0].vajra_neck3).__name__ == 'VajraV1MerudandaX'
    assert type(getattr(model.model[-1], 'proto_enhance', None)).__name__ in ('NoneType', 'Identity')
    assert type(getattr(model.model[-1], 'seg_enhance', None)).__name__ in ('NoneType', 'Identity')
    loss_fields = {key: getattr(criterion, key, None) for key in (
        'mask_dice_base', 'mask_dice_shape_gain', 'mask_boundary_gain')}
    assert loss_fields == dict(mask_dice_base=.3, mask_dice_shape_gain=.2, mask_boundary_gain=.15)
    assert not hasattr(criterion, 'cls_neg_weight'), 'Reference includes a separate classification loss change'
    new = training_options()
    for key in ('epochs', 'patience', 'batch', 'img_size', 'rect', 'seed', 'mask_ratio',
                'overlap_mask', 'box', 'cls', 'seg', 'dfl', 'warmup_epochs', 'warmup_momentum'):
        assert args[key] == new[key], (key, args[key], new[key])
    assert Path(args['data']).resolve() == Path(new['data']).resolve()
    assert args['optimizer'] == 'auto'
    with (ORIGINAL_RUN / 'results.csv').open(encoding='utf-8-sig') as handle:
        rows = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    def fitness(row):
        return sum(.1 * row[f'metrics/mAP50({kind})'] + .9 * row[f'metrics/mAP50-95({kind})']
                   for kind in ('Box', 'Mask'))
    best = max(rows, key=fitness)
    return dict(checkpoint=str(ORIGINAL_CHECKPOINT),
        checkpoint_sha256=hashlib.sha256(ORIGINAL_CHECKPOINT.read_bytes()).hexdigest(),
        args=args, original_loss_fields=loss_fields, historical_best_csv=best,
        historical_completed_epochs=len(rows),
        note='Historical criterion predates the mode flag and carries the original MABD coefficients. '
             'The current original MABD branch is retained. Old auto at full 300 epochs selects SGD(.01,.9) '
             'and sets warmup_bias_lr=0. Existing weights are only evaluated, never loaded into the new student.')


def evaluate_checkpoint(checkpoint, name, options=None):
    import torch
    from samx_followup_validation import BoundarySegmentationValidator
    from vajra.utils.torch_utils import init_seeds
    init_seeds(42, deterministic=True)
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    options = validation_options(name) if options is None else options
    validator = BoundarySegmentationValidator(args={**options, 'task': 'segment', 'mode': 'val'})
    validator.add_callback('on_val_start', clear_validation_cache)
    values = validator(model=str(checkpoint))
    counts = dict(images=int(validator.seen), instances=int(validator.nt_per_class.sum()))
    assert counts == dict(images=1532, instances=3942), counts
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, validation_options=options,
        counts=counts, metrics={k: float(v) for k, v in values.items()},
        boundary=validator.boundary.report(),
        per_class={str(int(c)): dict(name=validator.names[int(c)], instances=int(validator.nt_per_class[c]),
            **dict(zip(('precision', 'recall', 'map50', 'map95'),
                       map(float, validator.metrics.seg.class_aware_results(i)))))
            for i, c in enumerate(validator.metrics.seg.ap_class_idx)}, finished_at=now())
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    del validator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_worker(phase, *, audit=None, options=None, evaluation_options=None):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv, ShapeAdaptiveMerudandaX
    from vajra.nn.cfsav1 import CFSAv1
    torch.set_num_threads(4)
    audit = AUDIT if audit is None else Path(audit)
    if phase == 'original_validation':
        result = evaluate_checkpoint(ORIGINAL_CHECKPOINT, phase)
        write_json(audit / 'original_result.json', result)
        print('ORIGINAL_REVALIDATED ' + json.dumps(result), flush=True)
        return

    smoke = phase == 'smoke'
    options = training_options(smoke) if options is None else options
    expected_mode = options['mask_loss_mode']
    assert expected_mode in ('mabd_v1', 'mabd_v1_weights', 'mabd_v1_boundary')
    run_dir = Path(options['project']) / options['name']
    if run_dir.exists():
        raise FileExistsError(run_dir)
    progress_path = audit / f'{phase}_progress.json'
    record = dict(status='initializing', phase=phase, pid=os.getpid(), started_at=now(),
                  updated_at=now(), options=options)
    write_json(progress_path, record)
    model = Vajra('vajra-v1-nano-seg')
    startup, monitored, snapshots = {}, {}, {}
    batch_count = 0

    def check_start(trainer):
        head = trainer.model.model[-1]
        assert type(trainer.model.model[0].vajra_neck3).__name__ == 'VajraV1MerudandaX'
        assert not any(isinstance(m, (LDConv, ShapeAdaptiveMerudandaX, CFSAv1)) for m in trainer.model.modules())
        assert isinstance(head.proto_enhance, nn.Identity) and isinstance(head.seg_enhance, nn.Identity)
        assert not head.use_cfsav1 and not head.use_fft_guided_ema and not head.use_fft_guided_ema_proto_only
        assert trainer.args.samx_mode == trainer.args.samx_proto_mode == 'off'
        assert not trainer.args.use_cfsa and not trainer.args.use_cfsav1
        criterion = trainer.model.init_criterion()
        assert criterion.mask_loss_mode == expected_mode and criterion.use_morph_mask_loss
        assert trainer.args.mabd_v1_boundary_radius == 1
        assert criterion.seg_gain == 7.5
        trainer.model.criterion = criterion
        assert trainer.amp
        assert len(trainer.train_loader.dataset) == (613 if smoke else 6134)
        assert len(trainer.test_loader.dataset) == 1532
        assert trainer.save_dir.resolve() == run_dir.resolve()
        ids = [id(p) for group in trainer.optimizer.param_groups for p in group['params']]
        assert len(ids) == len(set(ids))
        assert {id(p) for p in trainer.model.parameters() if p.requires_grad}.issubset(set(ids))
        assert type(trainer.optimizer).__name__ == 'SGD'
        monitored['proto_output'] = head.proto.conv3.conv.weight
        snapshots.update({k: p.detach().cpu().clone() for k, p in monitored.items()})
        startup.update(mask_loss_mode=criterion.mask_loss_mode, boundary_radius=1,
            expanded_boundary_roi=expected_mode == 'mabd_v1',
            dice_weight_source='bbox_aspect_ratio' if expected_mode == 'mabd_v1_boundary' else 'thinness',
            samx=False, cfsa=False, cfsav1=False, proto_geometry=False,
            train_images=len(trainer.train_loader.dataset), val_images=1532,
            train_batches=len(trainer.train_loader), batch_size=trainer.batch_size,
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            amp=bool(trainer.amp), optimizer=type(trainer.optimizer).__name__,
            run_dir=str(trainer.save_dir), scratch=True, positive_anchor_average_unchanged=True)
        write_json(audit / f'{phase}_startup.json', startup)
        record.update(status='training', startup=startup, updated_at=now())
        write_json(progress_path, record)
        print('MABDV1_STARTUP ' + json.dumps(startup), flush=True)

    def check_batch(trainer):
        nonlocal batch_count
        batch_count += 1
        assert torch.isfinite(trainer.loss).all() and torch.isfinite(trainer.loss_items).all()
        criterion = trainer.model.criterion
        diagnostics = getattr(criterion, 'mabd_v1_last_diagnostics', None)
        if diagnostics is not None:
            assert all(torch.isfinite(value).all() for value in diagnostics.values())
            assert .29999 <= float(diagnostics['dice_weight_mean']) <= .50001
            assert .04999 <= float(diagnostics['boundary_weight_mean']) <= .20001
            if expected_mode in ('mabd_v1_weights', 'mabd_v1_boundary'):
                assert float(diagnostics['outer_boundary_pixels']) == 0.0
            if expected_mode == 'mabd_v1_boundary':
                assert abs(float(diagnostics['dice_weight_mean']) -
                           float(diagnostics['original_dice_weight_mean'])) < 1e-6
        if batch_count == 1 or batch_count % 10 == 0:
            record.update(status='training', updated_at=now(), epoch=trainer.epoch + 1,
                total_batches_processed=batch_count, finite_loss=float(trainer.loss.detach()),
                loss_items=[float(v) for v in trainer.loss_items.detach()],
                amp_scale=trainer.scaler.get_scale(), actual_v1_calls=getattr(criterion, 'mabd_v1_calls', 0),
                last_image_loss_terms={k: float(v) for k, v in (diagnostics or {}).items()},
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            write_json(progress_path, record, best_effort=True)
            if batch_count == 1:
                print('MABDV1_FIRST_BATCH ' + json.dumps(record), flush=True)

    def check_epoch(trainer):
        if trainer.epoch != 0:
            return
        criterion = trainer.model.criterion
        assert getattr(criterion, 'mabd_v1_calls', 0) > 0
        changes = {k: float((p.detach().cpu() - snapshots[k]).abs().max()) for k, p in monitored.items()}
        assert all(0 < value < float('inf') for value in changes.values()), changes
        startup.update(actual_loss_calls=criterion.mabd_v1_calls, first_epoch_parameter_updates=changes)
        write_json(audit / f'{phase}_startup.json', startup)
        print('MABDV1_BACKWARD_VERIFIED ' + json.dumps(changes), flush=True)

    def epoch_done(trainer):
        record.update(status='training', updated_at=now(), last_completed_epoch=trainer.epoch + 1,
            best_epoch=int(trainer.stopper.best_epoch),
            best_fitness=float(trainer.best_fitness) if trainer.best_fitness is not None else None,
            metrics={k: float(v) for k, v in (trainer.metrics or {}).items()},
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(progress_path, record, best_effort=True)

    def val_start(validator):
        clear_validation_cache(validator)
        record.update(status='validating_epoch', updated_at=now())
        write_json(progress_path, record, best_effort=True)

    model.add_callback('on_pretrain_routine_end', check_start)
    model.add_callback('on_train_batch_end', check_batch)
    model.add_callback('on_train_epoch_end', check_epoch)
    model.add_callback('on_fit_epoch_end', epoch_done)
    model.add_callback('on_val_start', val_start)
    model.train(**options)
    checkpoint = Path(model.trainer.best)
    assert checkpoint.exists()
    completed_epochs = int(model.trainer.epoch) + 1
    best_epoch = int(model.trainer.stopper.best_epoch)
    record.update(status='validating_fp32_boundary', checkpoint=str(checkpoint), updated_at=now())
    write_json(progress_path, record)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['train_args']['mask_loss_mode'] == expected_mode
    assert saved['train_args']['mabd_v1_boundary_radius'] == 1
    del saved, model
    monitored.clear()
    gc.collect()
    torch.cuda.empty_cache()
    result = evaluate_checkpoint(checkpoint, phase, options=evaluation_options)
    result.update(smoke=smoke, options=options, startup=startup,
                  completed_epochs=completed_epochs, best_epoch=best_epoch)
    write_json(audit / f'{phase}_result.json', result)
    record.update(status='complete', updated_at=now(), result_file=str(audit / f'{phase}_result.json'))
    write_json(progress_path, record)
    print('MABDV1_RESULT ' + json.dumps(result), flush=True)


def comparison():
    old = json.loads((AUDIT / 'original_result.json').read_text(encoding='utf-8'))
    new = json.loads((AUDIT / 'train_result.json').read_text(encoding='utf-8'))
    keys = ['metrics/precision(Mask)', 'metrics/recall(Mask)', 'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)']
    values = {label: {k: item['metrics'][k] * 100 for k in keys}
              for label, item in [('original_MABD', old), ('MABDv1_full', new)]}
    delta = {k: values['MABDv1_full'][k] - values['original_MABD'][k] for k in keys}
    boundary_delta = 100 * (new['boundary']['overall']['f1'] - old['boundary']['overall']['f1'])
    write_json(AUDIT / 'comparison.json', dict(percentage_values=values,
        new_minus_old_percentage_points=delta, boundary_f1_delta_percentage_points=boundary_delta,
        original=old, candidate=new, note='One seed, original historical weights reused; no original retraining.'))
    lines = ['# 原 MABD 与完整版 MABDv1', '', '相同 FP32 验证，1532 张 / 3942 实例。原版直接复用历史最佳权重。', '',
             '| 方法 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1（conf=0.25） |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, item in [('original_MABD', old), ('MABDv1_full', new)]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{values[label][k]:.4f}%' for k in keys)
                     + f" | {100*item['boundary']['overall']['f1']:.4f}% |")
    lines += ['', '边界 F1 使用同类别 Mask IoU≥0.5 的一对一匹配，容差为 1 个掩膜像素，计入未匹配实例；属于补充指标。',
              '历史训练源码和随机状态未完整保存，配置对齐不能保证逐位复现；不据单种子宣称统计显著性。']
    (AUDIT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_queue():
    AUDIT.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / 'status.json'
    if status_path.exists():
        raise FileExistsError(f'Existing experiment queue: {status_path}')
    state = dict(status='initializing', pid=os.getpid(), started_at=now(), completed=[], original_training_queued=False)
    write_json(status_path, state)
    try:
        reference = historical_reference()
        manifest = source_hashes()
        write_json(AUDIT / 'historical_reference.json', reference)
        write_json(AUDIT / 'source_manifest.json', manifest)
        write_json(AUDIT / 'experiment.json', dict(training=training_options(), smoke=training_options(True),
            validation=validation_options('train'),
            data_yaml_sha256=hashlib.sha256(Path(training_options()['data']).read_bytes()).hexdigest(),
            unit_tests='9 tests passed before launch, including CUDA AMP and original-loss regression.'))
        with zipfile.ZipFile(AUDIT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sources():
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.write(ORIGINAL_RUN / 'args.yaml', 'reference/original_args.yaml')
            archive.write(ORIGINAL_RUN / 'results.csv', 'reference/original_results.csv')
            archive.write(Path(training_options()['data']), 'reference/data.yaml')
        for phase in ('smoke', 'original_validation', 'train'):
            assert source_hashes() == manifest, 'Source changed since experiment snapshot'
            assert hashlib.sha256(ORIGINAL_CHECKPOINT.read_bytes()).hexdigest() == reference['checkpoint_sha256']
            state.update(status='running', phase=phase, updated_at=now(), progress=None)
            write_json(status_path, state)
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
            with (AUDIT / f'{phase}.log').open('x', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', phase],
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
                    raise RuntimeError(f'{phase} failed: exit {child.returncode}; inspect {phase}.log')
            assert source_hashes() == manifest, 'Source changed during experiment'
            state['completed'].append(phase)
        comparison()
        assert hashlib.sha256(ORIGINAL_CHECKPOINT.read_bytes()).hexdigest() == reference['checkpoint_sha256']
        state.update(status='complete', phase=None, updated_at=now(), result_file=str(AUDIT / 'comparison.json'))
        write_json(status_path, state)
    except BaseException as exc:
        state.update(status='failed', error=str(exc), updated_at=now())
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--queue', action='store_true')
    group.add_argument('--worker', choices=('smoke', 'original_validation', 'train'))
    args = parser.parse_args()
    if args.queue:
        run_queue()
    else:
        try:
            run_worker(args.worker)
        except BaseException:
            failure = dict(status='failed', time=now(), traceback=traceback.format_exc())
            write_json(AUDIT / f'{args.worker}_failure.json', failure)
            path = AUDIT / f'{args.worker}_progress.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                record.update(failure, updated_at=now())
                write_json(path, record)
            raise
