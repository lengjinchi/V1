"""Preflight, then train G1 and conditionally G2 in isolated child processes."""
import argparse
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

from run_samx_experiments import now, write_json
from run_samx_followups import SOURCES as PREVIOUS_SOURCES
from samx_branch_experiment_config import AUDIT, BASELINES, MODES, ROOT, decide_g2, options, validation_options

SOURCES = list(dict.fromkeys(PREVIOUS_SOURCES + [
    "vajra/nn/samx_branch_evidence.py", "vajra/nn/samx_mask_variants.py",
    "samx_branch_experiment_config.py", "samx_branch_validation.py", "samx_mask_experiment_config.py",
    "run_samx_branch_experiments.py", "tests/test_samx_branch_evidence.py",
    "docs/en/models/samx_branch_evidence.md",
]))


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def check_protected():
    records = json.loads((AUDIT/"protected_weights.json").read_text(encoding="utf-8"))
    for record in records:
        path = ROOT/record["path"]
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (record["size"], record["mtime_ns"]), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    return dict(checkpoints=len(records), verified_at=now())


def baseline_worker(mode):
    import torch
    from samx_followup_validation import validate_checkpoint
    torch.set_num_threads(4)
    previous = json.loads(BASELINES[mode].read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(previous["checkpoint"]).read_bytes()).hexdigest() == previous["checkpoint_sha256"]
    result = validate_checkpoint(previous["checkpoint"], validation_options("baseline_"+mode), expected=previous["metrics"])
    assert abs(result["boundary"]["overall"]["f1"]-previous["boundary"]["overall"]["f1"]) < 1e-10
    result.update(reproduced=True, finished_at=now())
    write_json(AUDIT/("baseline_"+mode+"_result.json"), result)


def worker(mode, smoke=False):
    import torch
    from torch import nn
    from vajra import Vajra
    from vajra.nn.modules import LDConv
    from vajra.nn.samx_branch_evidence import BranchEvidenceSAMX
    from vajra.nn.samx_followups import ProtoGeometryEnhance
    from samx_experiment_config import clear_validation_cache
    from samx_branch_validation import benchmark_pair, evaluate_evidence

    torch.set_num_threads(4)
    args = options(mode, smoke)
    prefix = ("smoke_" if smoke else "")+mode
    progress_path = AUDIT/(prefix+"_progress.json")
    run_dir = Path(args["project"])/args["name"]
    if progress_path.exists() or run_dir.exists():
        raise FileExistsError(f"Refusing duplicate or renamed experiment: {run_dir}")
    manifest = source_hashes()
    write_json(AUDIT/(prefix+"_source_manifest.json"), manifest)
    started, last_update = time.monotonic(), [0.]
    state = dict(status="initializing", mode=mode, smoke=smoke, pid=os.getpid(), started_at=now(),
                 updated_at=now(), options=args, last_completed_epoch=0)
    write_json(progress_path, state)
    model = Vajra("vajra-v1-nano-seg")
    startup, gradients, hooks = {}, dict(finite_nonzero_seen=False, calls=0, nonfinite_calls=0, zero_calls=0), []
    best_mask = dict(value=-1., epoch=0)

    def record_gradient(gradient):
        gradients["calls"] += 1
        if not torch.isfinite(gradient).all():
            gradients["nonfinite_calls"] += 1
        elif gradient.count_nonzero() > 0:
            gradients["finite_nonzero_seen"] = True
            gradients["first_finite_nonzero_call"] = gradients["calls"]
        else:
            gradients["zero_calls"] += 1

    def verify_start(trainer):
        neck, head = trainer.model.model[0].vajra_neck3, trainer.model.model[-1]
        assert type(neck) is BranchEvidenceSAMX
        assert neck.evidence_route[-1].weight.count_nonzero() == 0 and neck.evidence_route[-1].bias.count_nonzero() == 0
        assert sum(p.numel() for p in neck.evidence_route.parameters()) == 458
        assert isinstance(head.seg_enhance, nn.Identity)
        if mode == MODES[0]:
            assert isinstance(head.proto_enhance, nn.Identity)
        else:
            assert type(head.proto_enhance) is ProtoGeometryEnhance and head.proto_enhance.mode == "free"
            assert head.proto_enhance.hidden == 16
        for key in ("use_cfsa", "use_cfsav1", "cfsav1_global_context", "cfsav1_semantic_context", "cfsav1_gate_smooth",
                    "cfsav1_gate_ramp_epochs", "cfsav1_warmup_epochs", "samx_proto_share_coeff", "samx_proto_offset_bound"):
            assert not getattr(trainer.args, key), key
        assert not trainer.model.init_criterion().use_morph_mask_loss
        assert not trainer.resume and trainer.start_epoch == 0 and trainer.amp
        assert not trainer.args.pretrained and trainer.args.freeze is None
        assert trainer.epochs == (1 if smoke else 200) and trainer.args.patience == 30
        assert trainer.batch_size == 32 and trainer.args.img_size == 1600 and trainer.args.rect
        assert trainer.args.overlap_mask and trainer.args.mask_ratio == 4
        assert len(trainer.train_loader.dataset) == round(6134 * args["fraction"])
        assert len(trainer.test_loader.dataset) == 1532 and Path(trainer.save_dir) == run_dir
        samplers = [m for m in trainer.model.modules() if isinstance(m, LDConv)]
        assert len(samplers) == (2 if mode == MODES[0] else 3)
        for sampler in samplers:
            assert sampler.p_conv.weight.count_nonzero() == 0 and sampler.p_conv.bias.count_nonzero() == 0
        ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in trainer.model.parameters()}
        offset_ids = {id(p) for sampler in samplers for p in sampler.p_conv.parameters()}
        assert offset_ids == {id(p) for group in trainer.optimizer.param_groups if group.get("lr_scale") == .1 for p in group["params"]}
        startup.update(neck_class=type(neck).__name__, evidence_parameters=458, initial_correction_zero=True,
            shared_evidence_head=True, proto_mode=args["samx_proto_mode"],
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            train_images=len(trainer.train_loader.dataset), val_images=1532, amp=True,
            optimizer=type(trainer.optimizer).__name__, cfsa=False, cfsav1=False, mabd=False,
            batch=32, scratch_verified=True, offset_groups_verified=True, initial_amp_scale=float(trainer.scaler.get_scale()))
        hooks.append(neck.evidence_route[-1].weight.register_hook(record_gradient))
        state.update(status="training", startup=startup, updated_at=now())
        write_json(AUDIT/(prefix+"_startup.json"), startup)
        write_json(progress_path, state, best_effort=True)
        print("BRANCH_STARTUP_VERIFIED "+json.dumps(startup), flush=True)

    def batch_done(trainer):
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Non-finite training loss")
        if gradients["finite_nonzero_seen"] and hooks:
            for handle in hooks:
                handle.remove()
            hooks.clear()
        for group in trainer.optimizer.param_groups:
            if group.get("lr_scale") != .1:
                continue
            regular = [g for i, g in enumerate(trainer.optimizer.param_groups) if "lr_scale" not in g
                       and g["weight_decay"] == group["weight_decay"] and (i == 0) == group["warmup_bias"]]
            assert regular and abs(group["lr"]-.1*regular[0]["lr"]) < 1e-10
        startup["warmup_lr_ratio_verified"] = True
        if time.monotonic()-last_update[0] >= 10:
            last_update[0] = time.monotonic()
            state.update(status="training", updated_at=now(), current_epoch=trainer.epoch+1,
                loss=float(trainer.loss.detach()), gate_gradients=dict(gradients),
                amp_scale=float(trainer.scaler.get_scale()),
                elapsed_seconds=time.monotonic()-started, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(progress_path, state, best_effort=True)

    def val_start(validator):
        clear_validation_cache(validator)
        state.update(status="validating_epoch", updated_at=now())
        write_json(progress_path, state, best_effort=True)

    def save_mask_best(trainer):
        value = float(trainer.metrics.get("metrics/mAP50-95(Mask)", -1.))
        if value > best_mask["value"]:
            path = Path(trainer.weights_dir)/"best-mask-vajra-v1-nano-seg.pt"
            shutil.copyfile(trainer.last, path)
            best_mask.update(value=value, epoch=trainer.epoch+1, checkpoint=str(path))
            write_json(AUDIT/(prefix+"_mask_best.json"), best_mask)

    def epoch_done(trainer):
        startup["gate_gradient_verified"] = gradients["finite_nonzero_seen"]
        write_json(AUDIT/(prefix+"_startup.json"), startup)
        state.update(status="training", updated_at=now(), last_completed_epoch=trainer.epoch+1,
            best_epoch=int(trainer.stopper.best_epoch), best_fitness=float(trainer.best_fitness),
            metrics={k: float(v) for k, v in (trainer.metrics or {}).items()}, gate_gradients=dict(gradients),
            amp_scale=float(trainer.scaler.get_scale()),
            elapsed_seconds=time.monotonic()-started, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(progress_path, state, best_effort=True)
        print("BRANCH_EPOCH_COMPLETE "+json.dumps(state), flush=True)

    for event, callback in (("on_pretrain_routine_end", verify_start), ("on_train_batch_end", batch_done),
                            ("on_val_start", val_start), ("on_model_save", save_mask_best), ("on_fit_epoch_end", epoch_done)):
        model.add_callback(event, callback)
    try:
        model.train(**args)
    finally:
        for handle in hooks:
            handle.remove()
    assert gradients["finite_nonzero_seen"], "No finite nonzero gradient reached the added gate"
    trainer = model.trainer
    train_seconds = time.monotonic()-started
    peak = state["gpu_peak_allocated_gib"]
    state.update(status="validating_fp32_learned", updated_at=now())
    write_json(progress_path, state)
    result = evaluate_evidence(trainer.best, validation_options(prefix))
    result.update(mode=mode, smoke=smoke, options=args, startup=startup,
        completed_epochs=trainer.epoch+1, best_epoch=int(trainer.stopper.best_epoch),
        training_seconds=train_seconds, training_peak_allocated_gib=peak,
        mask_best_training_record=best_mask,
        selection="Unchanged combined Box+Mask fitness. Extra mask-best checkpoint is diagnostic only; not substituted.")
    write_json(AUDIT/(prefix+"_result.json"), result)
    state.update(status="validating_zero_evidence", updated_at=now())
    write_json(progress_path, state)
    zero = evaluate_evidence(trainer.best, validation_options(prefix+"_zero_evidence"), zero=True)
    write_json(AUDIT/(prefix+"_zero_evidence.json"), zero)
    baseline = json.loads(BASELINES[mode].read_text(encoding="utf-8"))
    state.update(status="benchmarking_pair", updated_at=now())
    write_json(progress_path, state)
    speed = benchmark_pair(baseline["checkpoint"], trainer.best)
    write_json(AUDIT/(prefix+"_paired_speed.json"), speed)
    result.update(forward_benchmark=speed["measurements"]["candidate"], paired_speed=speed,
        zero_evidence_metrics=zero["metrics"], source_unchanged=source_hashes() == manifest,
        finished_at=now(), elapsed_seconds=time.monotonic()-started)
    write_json(AUDIT/(prefix+"_result.json"), result)
    assert result["source_unchanged"], "Source changed during experiment; result retained"
    preserved = check_protected()
    state.update(status="complete", updated_at=now(), result_file=str(AUDIT/(prefix+"_result.json")), protected=preserved)
    write_json(progress_path, state)
    print("BRANCH_EXPERIMENT_COMPLETE "+json.dumps(result), flush=True)
    del model, trainer
    gc.collect(); torch.cuda.empty_cache()


def child(arguments, label, state, status_name):
    with (AUDIT/(label+".log")).open("x", encoding="utf-8") as logfile:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *arguments], cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            stdout=logfile, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state.update(worker_pid=process.pid, current_task=label, updated_at=now())
        state.pop("worker_progress", None)
        write_json(AUDIT/status_name, state)
        while process.poll() is None:
            path = AUDIT/(label+"_progress.json")
            if path.exists():
                try:
                    state["worker_progress"] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            state["updated_at"] = now()
            write_json(AUDIT/status_name, state, best_effort=True)
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"{label} failed with exit code {process.returncode}; inspect {label}.log")


def preflight():
    if (AUDIT/"preflight_status.json").exists():
        raise FileExistsError("Preflight already exists; refusing duplicate runs")
    tests = json.loads((AUDIT/"tests_result.json").read_text(encoding="utf-8"))
    manifest = source_hashes()
    assert tests["success"] and tests["source_manifest"] == manifest
    assert json.loads((ROOT/"output/samx_mask_experiments_20261030/status.json").read_text(encoding="utf-8"))["status"] == "complete"
    write_json(AUDIT/"source_manifest.json", manifest)
    with zipfile.ZipFile(AUDIT/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT/name, name)
    state = dict(status="running", phase="preflight", pid=os.getpid(), started_at=now(), completed=[])
    write_json(AUDIT/"preflight_status.json", state)
    try:
        check_protected()
        for mode in MODES:
            assert source_hashes() == manifest
            child(["--worker", mode, "--smoke"], "smoke_"+mode, state, "preflight_status.json")
            state["completed"].append("smoke_"+mode)
        for mode in MODES:
            assert source_hashes() == manifest
            child(["--baseline", mode], "baseline_"+mode, state, "preflight_status.json")
            state["completed"].append("baseline_"+mode)
        state.update(status="complete", updated_at=now(), current_task=None, source_manifest=manifest, protected=check_protected())
        write_json(AUDIT/"preflight_complete.json", state)
        write_json(AUDIT/"preflight_status.json", state)
    except BaseException:
        state.update(status="failed", updated_at=now(), traceback=traceback.format_exc())
        write_json(AUDIT/"preflight_status.json", state)
        raise


def update_summary():
    lines = ["# SAMX 分支结果门控实验", "", "旧模块和权重保留。G1 先训练，按预定条件决定 G2；未完成结果不填数。", "",
             "| 方法 | Mask P | Mask R | Mask AP50 | Mask AP50:95 | 边界 F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    records = {}
    keys = ("metrics/precision(Mask)", "metrics/recall(Mask)", "metrics/mAP50(Mask)", "metrics/mAP50-95(Mask)")
    for mode in MODES:
        baseline = json.loads(BASELINES[mode].read_text(encoding="utf-8"))
        candidates = [("对照_"+mode, baseline)]
        path = AUDIT/(mode+"_result.json")
        progress_path = AUDIT/(mode+"_progress.json")
        if path.exists() and progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8"))["status"] == "complete":
            candidate = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((mode, candidate)); records[mode] = candidate
        for label, result in candidates:
            values = [f"{100*result['metrics'][key]:.2f}%" for key in keys]
            lines.append("| "+label+" | "+" | ".join(values)+f" | {100*result['boundary']['overall']['f1']:.2f}% |")
    decision = AUDIT/"g2_decision.json"
    if decision.exists():
        data = json.loads(decision.read_text(encoding="utf-8"))
        lines.extend(["", "G2 决策："+("达到条件，继续训练。" if data["run_g2"] else "未达到预定整体收益条件，跳过。"), "", "```json", json.dumps(data, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "正式结果为综合 fitness 最佳权重的完整 FP32 验证；额外保留的 Mask 单项最佳权重不混入主表。", "各模型同进程交替测速、门控统计及关闭新增修正器的干预结果见对应 JSON。单个种子的变化不代表统计显著。"])
    (AUDIT/"结果汇总.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(AUDIT/"comparison.json", records)


def queue():
    if (AUDIT/"status.json").exists():
        raise FileExistsError("Queue already exists; refusing duplicate training")
    pre = json.loads((AUDIT/"preflight_complete.json").read_text(encoding="utf-8"))
    assert pre["status"] == "complete" and pre["source_manifest"] == source_hashes()
    state = dict(status="running", phase="training", pid=os.getpid(), started_at=now(), updated_at=now(),
                 modes=list(MODES), completed=[], skipped=[])
    write_json(AUDIT/"status.json", state)
    try:
        check_protected(); update_summary()
        child(["--worker", MODES[0]], MODES[0], state, "status.json")
        state["completed"].append(MODES[0])
        baseline = json.loads(BASELINES[MODES[0]].read_text(encoding="utf-8"))
        result = json.loads((AUDIT/(MODES[0]+"_result.json")).read_text(encoding="utf-8"))
        assert result["source_unchanged"] and source_hashes() == pre["source_manifest"]
        decision = decide_g2(baseline, result)
        decision["decided_at"] = now()
        write_json(AUDIT/"g2_decision.json", decision)
        update_summary()
        if decision["run_g2"]:
            check_protected()
            child(["--worker", MODES[1]], MODES[1], state, "status.json")
            state["completed"].append(MODES[1])
        else:
            state["skipped"].append(dict(mode=MODES[1], reason="G1 did not meet predeclared overall-improvement thresholds"))
        assert source_hashes() == pre["source_manifest"]
        state.update(status="complete", current_task=None, updated_at=now(), protected=check_protected())
        write_json(AUDIT/"status.json", state)
        update_summary()
    except BaseException:
        state.update(status="failed", updated_at=now(), traceback=traceback.format_exc())
        write_json(AUDIT/"status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--queue", action="store_true")
    group.add_argument("--baseline", choices=MODES)
    group.add_argument("--worker", choices=MODES)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke and not arguments.worker:
        parser.error("--smoke requires --worker")
    try:
        if arguments.preflight: preflight()
        elif arguments.queue: queue()
        elif arguments.baseline: baseline_worker(arguments.baseline)
        else: worker(arguments.worker, arguments.smoke)
    except BaseException:
        label = ("smoke_" if arguments.smoke else "")+(arguments.worker or ("baseline_"+arguments.baseline if arguments.baseline else "preflight" if arguments.preflight else "queue"))
        failure = dict(status="failed", updated_at=now(), pid=os.getpid(), traceback=traceback.format_exc())
        write_json(AUDIT/(label+"_failure.json"), failure)
        progress = AUDIT/(label+"_progress.json")
        if progress.exists():
            state = json.loads(progress.read_text(encoding="utf-8")); state.update(failure); write_json(progress, state)
        raise
