"""Train original-Dice/thinness-boundary ablation; preserve all three references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from mabd_boundary_experiment_config import (
    ROOT, AUDIT, WEIGHTS_AUDIT, training_options, validation_options,
)
from run_mabd_experiment import run_worker
from run_mabd_weights_experiment import sources as previous_sources, verified_references as previous_references
from run_samx_experiments import now, write_json


def sources():
    return sorted(set(previous_sources() + [ROOT / name for name in (
        'mabd_boundary_experiment_config.py', 'run_mabd_boundary_experiment.py',
        'docs/en/models/mabd_v1_boundary_experiment.md',
    )]))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}


def verified_references():
    references = previous_references()
    weights = json.loads((WEIGHTS_AUDIT / 'train_result.json').read_text(encoding='utf-8'))
    assert weights['counts'] == {'images': 1532, 'instances': 3942}
    assert hashlib.sha256(Path(weights['checkpoint']).read_bytes()).hexdigest() == weights['checkpoint_sha256']
    expected = {k: v for k, v in validation_options('train').items() if k not in ('project', 'name')}
    actual = {k: v for k, v in weights['validation_options'].items() if k not in ('project', 'name')}
    assert actual == expected
    assert weights['boundary']['protocol'] == references['original_MABD']['boundary']['protocol']
    candidate = training_options()
    diff = {key for key in weights['options'].keys() | candidate.keys()
            if weights['options'].get(key) != candidate.get(key)}
    assert diff == {'name', 'mask_loss_mode'}, diff
    references['MABDv1_weights_only'] = weights
    return references


def comparison(references):
    candidate = json.loads((AUDIT / 'train_result.json').read_text(encoding='utf-8'))
    label = 'MABDv1_original_dice_thinness_boundary'
    rows = {**references, label: candidate}
    keys = ('metrics/precision(Mask)', 'metrics/recall(Mask)',
            'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)')
    values = {name: {**{key: item['metrics'][key] * 100 for key in keys},
                     'boundary_f1': item['boundary']['overall']['f1'] * 100}
              for name, item in rows.items()}
    differences = {name: {key: values[label][key] - value for key, value in baseline.items()}
                   for name, baseline in values.items() if name != label}
    assert candidate['counts'] == references['original_MABD']['counts']
    assert candidate['boundary']['protocol'] == references['original_MABD']['boundary']['protocol']
    write_json(AUDIT / 'comparison.json', dict(
        percentage_values=values, candidate_minus_reference_percentage_points=differences, results=rows,
        note='One seed. Only the original-Dice/thinness-boundary candidate was trained; '
             'three reference results reused after hash and protocol checks.'))
    labels = {'original_MABD': '原 MABD', 'MABDv1_full': '完整版：thinness 双权重 + 扩张 ROI',
              'MABDv1_weights_only': '上一轮：thinness 双权重 + 原 GT ROI',
              label: '本次：原 Dice + thinness 边界 + 原 GT ROI'}
    lines = ['# MABDv1：恢复原 Dice 权重的单项消融', '',
             '统一 FP32 验证：1532 张、3942 个实例；三组历史结果及权重均已核对。', '',
             '| 方法 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in values.items():
        lines.append('| ' + labels[name] + ' | ' + ' | '.join(f'{v:.4f}%' for v in row.values()) + ' |')
    lines += ['', '与上一轮只改变 Dice 权重来源：由 thinness 恢复为原框长宽比因子。',
              '边界权重仍为 0.05+0.15×thinness；边界带、GT ROI、FP32 计算及正样本平均不变。',
              '边界 F1：conf=0.25、同类 Mask IoU≥0.5 贪心一对一匹配、1 个掩膜像素容差，计入未匹配实例。',
              '单种子结果用于判断趋势，不能证明统计显著或直接确认因果。']
    (AUDIT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_queue():
    AUDIT.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / 'status.json'
    if status_path.exists():
        raise FileExistsError(f'Existing queue: {status_path}')
    state = dict(status='initializing', pid=os.getpid(), started_at=now(), completed=[],
                 historical_training_queued=False, only_candidate='mabd_v1_boundary')
    write_json(status_path, state)
    try:
        references = verified_references()
        manifest = source_hashes()
        write_json(AUDIT / 'references.json', references)
        write_json(AUDIT / 'source_manifest.json', manifest)
        write_json(AUDIT / 'experiment.json', dict(
            training=training_options(), smoke=training_options(True), validation=validation_options('train'),
            difference_from_previous='Restore original morphology_factors Dice weight; keep boundary and GT ROI.',
            reference_audit=str(WEIGHTS_AUDIT),
            data_yaml_sha256=hashlib.sha256(Path(training_options()['data']).read_bytes()).hexdigest(),
            unit_tests='18 passed before launch, including original-Dice equivalence, single-change checks and CUDA AMP.'))
        with zipfile.ZipFile(AUDIT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sources():
                archive.write(path, path.relative_to(ROOT).as_posix())
            for filename in ('source_manifest.json', 'train_result.json', 'experiment.json', 'references.json'):
                archive.write(WEIGHTS_AUDIT / filename, 'reference/' + filename)
            archive.write(Path(training_options()['data']), 'reference/data.yaml')
        for phase in ('smoke', 'train'):
            assert source_hashes() == manifest, 'Source changed since snapshot'
            verified_references()
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
            state['progress'] = json.loads((AUDIT / f'{phase}_progress.json').read_text(encoding='utf-8'))
            state['completed'].append(phase)
            write_json(status_path, state)
        verified_references()
        comparison(references)
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
    group.add_argument('--worker', choices=('smoke', 'train'))
    args = parser.parse_args()
    if args.queue:
        run_queue()
    else:
        try:
            run_worker(args.worker, audit=AUDIT, options=training_options(args.worker == 'smoke'),
                       evaluation_options=validation_options(args.worker))
        except BaseException:
            failure = dict(status='failed', time=now(), traceback=traceback.format_exc())
            write_json(AUDIT / f'{args.worker}_failure.json', failure)
            progress_path = AUDIT / f'{args.worker}_progress.json'
            if progress_path.exists():
                record = json.loads(progress_path.read_text(encoding='utf-8'))
                record.update(failure, updated_at=now())
                write_json(progress_path, record)
            raise
