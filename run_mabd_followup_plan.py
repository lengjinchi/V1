"""Durable conditional execution: previous run -> diagnosis -> E1 -> gated E2/repeats/joint."""
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

from mabd_followup_config import (
    ROOT,AUDIT,PREVIOUS_AUDIT,REPEAT_SEEDS,SCREEN,SCALE_MISMATCH,
    training_options,validation_options,candidate_passes,
)
from run_mabd_boundary_experiment import sources as previous_sources, verified_references as historical_references
from run_samx_experiments import now,write_json


def sources():
    return sorted(set(previous_sources()+[ROOT/name for name in (
        'mabd_followup_config.py','diagnose_mabd_followups.py','run_mabd_followup_worker.py',
        'run_mabd_followup_plan.py','tests/test_mabd_followup_plan.py',
        'joint_experiment_config.py','run_joint_experiment.py',
        'docs/en/models/mabd_next_plan_20261110.md')]))


def source_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}


def references():
    refs=historical_references()
    current=json.loads((PREVIOUS_AUDIT/'train_result.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(current['checkpoint']).read_bytes()).hexdigest()==current['checkpoint_sha256']
    assert current['counts']==refs['original_MABD']['counts']
    assert current['boundary']['protocol']==refs['original_MABD']['boundary']['protocol']
    expected={k:v for k,v in validation_options('reference').items() if k not in ('project','name')}
    actual={k:v for k,v in current['validation_options'].items() if k not in ('project','name')}
    assert expected==actual
    refs['MABDv1_original_dice_thinness_boundary']=current
    return refs


def render_comparison(refs,results):
    rows={**refs,**results}
    keys=('metrics/precision(Mask)','metrics/recall(Mask)','metrics/mAP50(Mask)','metrics/mAP50-95(Mask)')
    values={label:{**{key:item['metrics'][key]*100 for key in keys},
                   'boundary_f1':item['boundary']['overall']['f1']*100} for label,item in rows.items()}
    write_json(AUDIT/'comparison.json',dict(percentage_values=values,results=rows,
        note='Historical seed42 reused. Newly trained repeat/joint pairs share this source snapshot. '
             'Validation screening only; no significance claim and no test-set selection.'))
    lines=['# MABD 后续计划结果','',
        'FP32、1532 张验证图像、3942 个实例；边界 F1 使用固定 conf=0.25 和已记录的一对一匹配协议。','',
        '| 实验 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,row in values.items():
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}%' for v in row.values())+' |')
    lines+=['','历史 seed42 及源码复现限制见 references.json；新增种子成对比较，不能把筛选门槛视作显著性检验。',
            '联合实验使用成对新训练的原 MABD/优选损失，两者均为同一结构和 300 轮上限。历史联合实验上限 200 轮，单独作为背景记录。']
    (AUDIT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_plan():
    AUDIT.mkdir(parents=True,exist_ok=True)
    status_path=AUDIT/'status.json'
    if status_path.exists():
        raise FileExistsError(f'Refusing duplicate plan launch: {status_path}')
    state=dict(status='waiting_previous',pid=os.getpid(),started_at=now(),updated_at=now(),completed=[],
        current_phase='waiting_previous',decisions=[],authorized_plan='User: 这个实验跑完按这个计划执行',
        max_new_training_jobs=8,repeat_seeds=list(REPEAT_SEEDS))
    write_json(status_path,state)
    results={}
    try:
        while True:
            previous=json.loads((PREVIOUS_AUDIT/'status.json').read_text(encoding='utf-8'))
            if previous['status']=='complete':
                break
            if previous['status']=='failed':
                raise RuntimeError('Previous experiment failed; follow-ups were not started')
            import psutil
            if not psutil.pid_exists(previous['pid']):
                raise RuntimeError('Previous queue ended without a final result')
            state.update(updated_at=now(),previous_progress=previous.get('progress'))
            write_json(status_path,state,best_effort=True)
            time.sleep(15)
        refs=references()
        manifest=source_hashes()
        write_json(AUDIT/'references.json',refs)
        write_json(AUDIT/'source_manifest.json',manifest)
        write_json(AUDIT/'execution_plan.json',dict(
            screening_thresholds=SCREEN,scale_mismatch_threshold=SCALE_MISMATCH,
            repeats=list(REPEAT_SEEDS),e1=training_options('E1_noBase'),
            e2_rule='Only if E1 fails screening AND train-only calibration is eligible; one bounded calibration.',
            repeat_rule='Only a candidate that passes screening; paired original and candidate at seeds 0 and 123.',
            joint_rule='Only if both new seed pairs preserve AP direction and stability; paired original/candidate joint runs.',
            maximum_formal_training_jobs=8,unit_tests='25 passed before launch, including CUDA AMP and historical-loss regression.',
            old_seed42_retraining=False,source_mutations_after_launch=False,
            previous_final_result=str(PREVIOUS_AUDIT/'train_result.json')))
        with zipfile.ZipFile(AUDIT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for path in sources():
                archive.write(path,path.relative_to(ROOT).as_posix())
            archive.write(PREVIOUS_AUDIT/'train_result.json','reference/latest_result.json')
            archive.write(Path(training_options('E1_noBase')['data']),'reference/data.yaml')

        def run_process(command,log_path,phase,progress_path=None):
            assert source_hashes()==manifest,'Source changed after plan snapshot'
            references()
            state.update(status='running',current_phase=phase,updated_at=now(),progress=None)
            write_json(status_path,state)
            env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
            with log_path.open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
                state['worker_pid']=child.pid
                while child.poll() is None:
                    if progress_path is not None and progress_path.exists():
                        try:
                            state['progress']=json.loads(progress_path.read_text(encoding='utf-8'))
                        except (OSError,json.JSONDecodeError):
                            pass
                    state['updated_at']=now();write_json(status_path,state,best_effort=True)
                    time.sleep(3)
                if child.returncode:
                    raise RuntimeError(f'{phase} failed ({child.returncode}); inspect {log_path}')
            assert source_hashes()==manifest,'Source changed during plan worker'
            state['completed'].append(phase)
            write_json(status_path,state)

        run_process([sys.executable,'-u',str(Path(__file__).resolve()),'--diagnose'],
            AUDIT/'diagnostics.log','diagnostics',AUDIT/'diagnostics_progress.json')
        diagnostic=json.loads((AUDIT/'weight_diagnostics.json').read_text(encoding='utf-8'))

        def job(**spec):
            tag=spec['tag'];directory=AUDIT/'jobs'/tag
            directory.mkdir(parents=True,exist_ok=False)
            job_path=directory/'job.json';write_json(job_path,spec)
            for phase in ('smoke','train'):
                command=[sys.executable,'-u',str(ROOT/'run_mabd_followup_worker.py'),'--job',str(job_path)]
                if phase=='smoke':
                    command.append('--smoke')
                run_process(command,directory/f'{phase}.log',f'{tag}:{phase}',directory/f'{phase}_progress.json')
            result=json.loads((directory/'train_result.json').read_text(encoding='utf-8'))
            results[tag]=result;render_comparison(refs,results)
            return result

        def decision(stage,details):
            item=dict(stage=stage,time=now(),**details)
            state['decisions'].append(item);write_json(status_path,state)
            write_json(AUDIT/'decisions.json',state['decisions'])
            print('PLAN_DECISION '+json.dumps(item),flush=True)

        baseline=refs['original_MABD']
        e1=job(tag='E1_noBase',seed=42,gain=.15)
        e1_gate=candidate_passes(e1,baseline)
        decision('E1_screening',e1_gate)
        chosen,chosen_gain,chosen_tag=(e1,.15,'E1_noBase') if e1_gate['passed'] else (None,None,None)
        calibration=diagnostic['calibration']
        if chosen is None and calibration['eligible'] and abs(calibration['gain']-.15)>.0015:
            e2=job(tag='E2_calibrated',seed=42,gain=calibration['gain'])
            e2_gate=candidate_passes(e2,baseline);decision('E2_screening',e2_gate)
            if e2_gate['passed']:
                chosen,chosen_gain,chosen_tag=e2,calibration['gain'],'E2_calibrated'
        else:
            decision('E2_not_queued',dict(reason='E1 passed, or no eligible nonduplicate train-only calibration',
                calibration=calibration))
        if chosen is None:
            decision('stop_after_ablations',dict(reason='No candidate passed the prespecified screening; retain original MABD.',
                repeats_queued=False,joint_queued=False))
        else:
            paired=[]
            for seed in REPEAT_SEEDS:
                original=job(tag=f'original_seed{seed}',seed=seed,original=True)
                candidate=job(tag=f'{chosen_tag}_seed{seed}',seed=seed,gain=chosen_gain)
                gate=candidate_passes(candidate,original)
                # Repeat consistency: no requirement for another >=0.3pp gain,
                # but neither AP may decrease; all other stability checks hold.
                checks={k:v for k,v in gate['checks'].items() if k!='at_least_one_ap_gain'}
                checks['both_aps_nonnegative']=all(gate['metric_deltas'][key]>=0 for key in (
                    'metrics/mAP50(Mask)','metrics/mAP50-95(Mask)'))
                paired.append(dict(seed=seed,passed=all(checks.values()),checks=checks,details=gate))
                decision(f'repeat_seed{seed}',paired[-1])
            write_json(AUDIT/'repeat_summary.json',dict(pairs=paired,
                historical_seed42_gate=candidate_passes(chosen,baseline),
                note='Two fresh paired seeds plus a historical seed42 reference; descriptive, not a significance claim.'))
            if all(pair['passed'] for pair in paired):
                original_joint=job(tag='joint_original',seed=42,original=True,joint=True)
                candidate_joint=job(tag=f'joint_{chosen_tag}',seed=42,gain=chosen_gain,joint=True)
                decision('joint_comparison',candidate_passes(candidate_joint,original_joint))
                historical=ROOT/'output/joint_cfsav1_samx_20261109/train_result.json'
                if historical.exists():
                    write_json(AUDIT/'historical_joint_context.json',dict(
                        result=json.loads(historical.read_text(encoding='utf-8')),
                        warning='Historical BCE joint run used 200-epoch horizon; new joint pair uses 300. Context only.'))
            else:
                decision('joint_not_queued',dict(reason='Fresh paired seeds did not consistently preserve both AP metrics.'))
        references();render_comparison(refs,results)
        state.update(status='complete',current_phase=None,updated_at=now(),
            result_file=str(AUDIT/'comparison.json'),selected_candidate=chosen_tag,
            conclusion='See decisions.json; no automatic promotion or deletion of original methods.')
        write_json(status_path,state)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc(),updated_at=now())
        write_json(status_path,state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--queue',action='store_true')
    group.add_argument('--diagnose',action='store_true')
    args=parser.parse_args()
    if args.queue:
        run_plan()
    else:
        from diagnose_mabd_followups import run_diagnostics
        run_diagnostics(references())
