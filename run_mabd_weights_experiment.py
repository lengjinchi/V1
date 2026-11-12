"""One ablation only: smoke -> train -> FP32/boundary comparison.

Historical original MABD and full MABDv1 results/weights are read and verified;
neither is retrained or overwritten. No further experiment is queued.
"""
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

from mabd_weights_experiment_config import (
    ROOT, AUDIT, FULL_AUDIT, training_options, validation_options,
)
from run_mabd_experiment import run_worker, sources as shared_sources
from run_samx_experiments import now, write_json


def sources():
    return sorted(set(shared_sources() + [ROOT / name for name in (
        'mabd_weights_experiment_config.py', 'run_mabd_weights_experiment.py',
        'docs/en/models/mabd_v1_weights_experiment.md',
    )]))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}


def verified_references():
    references = {}
    for label, filename in [('original_MABD', 'original_result.json'), ('MABDv1_full', 'train_result.json')]:
        result = json.loads((FULL_AUDIT / filename).read_text(encoding='utf-8'))
        assert result['counts'] == {'images': 1532, 'instances': 3942}
        digest = hashlib.sha256(Path(result['checkpoint']).read_bytes()).hexdigest()
        assert digest == result['checkpoint_sha256'], f'Changed reference: {label}'
        expected = validation_options('train')
        for key in ('project', 'name'):
            expected.pop(key)
        actual = {k: v for k, v in result['validation_options'].items() if k not in ('project', 'name')}
        assert actual == expected, (label, actual, expected)
        references[label] = result
    full_options = references['MABDv1_full']['options']
    candidate = training_options()
    diff = {key for key in full_options.keys() | candidate.keys() if full_options.get(key) != candidate.get(key)}
    assert diff == {'name', 'mask_loss_mode'}, diff
    prior = json.loads((FULL_AUDIT / 'experiment.json').read_text(encoding='utf-8'))
    data_digest = hashlib.sha256(Path(candidate['data']).read_bytes()).hexdigest()
    assert data_digest == prior['data_yaml_sha256']
    assert references['original_MABD']['boundary']['protocol'] == references['MABDv1_full']['boundary']['protocol']
    return references


def comparison(references):
    candidate = json.loads((AUDIT / 'train_result.json').read_text(encoding='utf-8'))
    rows = {**references, 'MABDv1_weights_only': candidate}
    keys = ('metrics/precision(Mask)', 'metrics/recall(Mask)',
            'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)')
    values = {label: {**{key: item['metrics'][key] * 100 for key in keys},
                      'boundary_f1': item['boundary']['overall']['f1'] * 100}
              for label, item in rows.items()}
    differences = {label: {key: values['MABDv1_weights_only'][key] - value for key, value in baseline.items()}
                   for label, baseline in values.items() if label != 'MABDv1_weights_only'}
    assert candidate['counts'] == references['original_MABD']['counts']
    assert candidate['boundary']['protocol'] == references['original_MABD']['boundary']['protocol']
    write_json(AUDIT / 'comparison.json', dict(
        percentage_values=values, candidate_minus_reference_percentage_points=differences,
        results=rows, note='One seed; historical original/full results reused after hash and protocol checks. '
                           'Only the weights-only variant was trained in this queue.'))
    lines = ['# MABDv1 单项消融：只改形态权重', '',
             '统一 FP32 验证：1532 张、3942 个实例。原 MABD 与完整版复用已核对哈希的结果。', '',
             '| 方法 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    labels = {'original_MABD': '原 MABD', 'MABDv1_full': '完整版：thinness + 扩张 ROI',
              'MABDv1_weights_only': '本次：thinness + 原 GT ROI'}
    for label, row in values.items():
        lines.append('| ' + labels[label] + ' | ' + ' | '.join(f'{v:.4f}%' for v in row.values()) + ' |')
    lines += ['', '边界 F1 使用 conf=0.25、同类 Mask IoU≥0.5 贪心一对一匹配、1 个掩膜像素容差，计入未匹配实例。',
              '本次与完整版仅改变边界 ROI；BCE/Dice 支持区域、形态学半径、动态权重和正样本平均均保持一致。',
              '单种子消融用于判断趋势；不能据此声称统计显著或将差异全部归因于 ROI。']
    (AUDIT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_queue():
    AUDIT.mkdir(parents=True, exist_ok=True)
    status_path = AUDIT / 'status.json'
    if status_path.exists():
        raise FileExistsError(f'Existing queue: {status_path}')
    state = dict(status='initializing', pid=os.getpid(), started_at=now(), completed=[],
                 original_training_queued=False, full_training_queued=False,
                 only_candidate='mabd_v1_weights')
    write_json(status_path, state)
    try:
        references = verified_references()
        manifest = source_hashes()
        write_json(AUDIT / 'references.json', references)
        write_json(AUDIT / 'source_manifest.json', manifest)
        write_json(AUDIT / 'experiment.json', dict(
            training=training_options(), smoke=training_options(True), validation=validation_options('train'),
            difference_from_full='Original GT boundary ROI; radius=1 and thinness weights unchanged.',
            reference_audit=str(FULL_AUDIT),
            data_yaml_sha256=hashlib.sha256(Path(training_options()['data']).read_bytes()).hexdigest(),
            unit_tests='13 passed before launch, including ROI gradients, original/full regression and CUDA AMP.'))
        with zipfile.ZipFile(AUDIT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sources():
                archive.write(path, path.relative_to(ROOT).as_posix())
            for filename in ('source_manifest.json', 'original_result.json', 'train_result.json', 'experiment.json'):
                archive.write(FULL_AUDIT / filename, 'reference/' + filename)
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
