"""Validate one checkpoint with learned, zero, 0.1 and 0.25 residual gains.

Only temporary inference hooks change outputs. No parameter/checkpoint is edited.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from diagnose_cfsav1_regions import valid_mask
from run_samx_experiments import now, write_json
from vajra.models.vajra.segment.val import SegmentationValidator
from vajra.nn.samx_geometry import GeometricResidualBlock, GeometryResidualSAMX
from vajra.utils.torch_utils import init_seeds


ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/samx_geometry_diagnostic_20261029'
SOURCE_RESULT = ROOT/'output/samx_geometry_20261028/train_result.json'
MODES = {'learned': None, 'zero': 0., 'gain0.1': .1, 'gain0.25': .25}
MASK_KEYS = ['metrics/precision(Mask)', 'metrics/recall(Mask)',
             'metrics/mAP50(Mask)', 'metrics/mAP50-95(Mask)']


class GainIntervention:
    """Observe branch amplitudes, apply a temporary scalar, remove all hooks."""
    def __init__(self, model, mode='learned'):
        if mode not in MODES:
            raise ValueError(mode)
        self.model, self.mode = model, mode
        self.handles, self.pending, self.statistics = [], {}, {}
        self.batch, self.mask_cache = None, {}

    def set_batch(self, batch):
        self.batch, self.mask_cache = batch, {}

    def __enter__(self):
        try:
            for name, block in self.model.named_modules():
                if not isinstance(block, GeometricResidualBlock):
                    continue
                self.statistics[name] = dict(elements=0, main_squared_sum=0.,
                    raw_squared_sum=0., correction_squared_sum=0., cross_sum=0.,
                    trained_gain=float(block.geometry_scale.weight.detach().float()),
                    applied_gain=(float(block.geometry_scale.weight.detach().float())
                                  if MODES[self.mode] is None else MODES[self.mode]))
                self.handles.append(block.regular_head.register_forward_hook(self.main_hook(name)))
                self.handles.append(block.geometry_scale.register_forward_hook(self.scale_hook(name)))
            if len(self.statistics) != 2:
                raise ValueError(f'Expected two geometry stages; got {list(self.statistics)}')
            return self
        except BaseException:
            self.close()
            raise

    def main_hook(self, name):
        def hook(module, args, output):
            self.pending[name] = output.detach()
        return hook

    def scale_hook(self, name):
        def hook(module, args, output):
            target = MODES[self.mode]
            applied = output if target is None else args[0]*args[0].new_tensor(target)
            main = self.pending.pop(name).float()
            raw, correction = args[0].detach().float(), applied.detach().float()
            h, w = main.shape[-2:]
            if (h,w) not in self.mask_cache:
                self.mask_cache[h,w] = (valid_mask(self.batch,(h,w)) if self.batch is not None
                    else torch.ones(main.shape[0],1,h,w,device=main.device,dtype=torch.bool))
            valid = self.mask_cache[h,w]
            values = torch.stack(((main.square()*valid).sum(),
                                  (raw.square()*valid).sum(),
                                  (correction.square()*valid).sum(),
                                  (main*correction*valid).sum(),valid.sum())).cpu().tolist()
            s = self.statistics[name]
            for key, value in zip(('main_squared_sum','raw_squared_sum','correction_squared_sum','cross_sum'),values):
                s[key] += value
            s['elements'] += int(values[4])*main.shape[1]
            return None if target is None else applied
        return hook

    def report(self):
        stages = {}
        for name,s in self.statistics.items():
            if not s['elements'] or not s['main_squared_sum']:
                raise ValueError(f'No valid response statistics for {name}')
            main, raw, corr = (s[k] for k in ('main_squared_sum','raw_squared_sum','correction_squared_sum'))
            stages[name] = dict(trained_gain=s['trained_gain'],applied_gain=s['applied_gain'],
                elements=s['elements'],main_rms=math.sqrt(main/s['elements']),
                raw_geometry_rms=math.sqrt(raw/s['elements']),correction_rms=math.sqrt(corr/s['elements']),
                correction_to_main_rms=math.sqrt(corr/main),
                raw_geometry_to_main_rms=math.sqrt(raw/main),
                main_correction_cosine=s['cross_sum']/math.sqrt(main*corr) if corr else None)
        return dict(stages=stages,
            note='RMS statistics exclude padding anchor cells; gamma0 interventions do not skip branch computation. Both stages change together; stage3 inputs may therefore also change. Observed timing includes hook work and is not a deployment speed benchmark.')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        self.mask_cache.clear()
        self.batch = None

    def __exit__(self,*args):
        self.close()


class GeometryValidator(SegmentationValidator):
    def __init__(self,*args,intervention='learned',**kwargs):
        super().__init__(*args,**kwargs)
        self.intervention, self.observer = intervention, None

    def init_metrics(self,model):
        super().init_metrics(model)
        self.observer = GainIntervention(model,self.intervention).__enter__()

    def preprocess(self,batch):
        batch = super().preprocess(batch)
        self.observer.set_batch(batch)
        return batch

    def __call__(self,*args,**kwargs):
        try:
            return super().__call__(*args,**kwargs)
        finally:
            if self.observer:
                self.observer.close()


def self_check():
    """Check actual intervention outputs, recovery, parameters and padding count."""
    torch.manual_seed(42)
    module = GeometryResidualSAMX(8,8,8,4,rep_csp=True).eval()
    with torch.no_grad():
        for i,block in enumerate((module.block2,module.block3)):
            block.geometry.p_conv.bias.uniform_(-.7,.7)
            block.geometry_scale.weight.fill_(.6+.2*i)
    x = torch.randn(2,8,7,11)
    state = {k:v.clone() for k,v in module.state_dict().items()}
    def manual(gain):
        f1,f2 = module.conv1(x).chunk(2,1)
        ys = [f1,f2]
        for block in (module.block2,module.block3):
            t = block.shared(ys[-1])
            ys.append(block.regular_head(t)+gain*block.geometry(t))
        return module.conv4(torch.cat(ys,1))
    with torch.inference_mode():
        original = module(x)
        for mode, gain in MODES.items():
            reference = original if gain is None else manual(gain)
            with GainIntervention(module,mode) as observer:
                actual = module(x)
                torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                report = observer.report()
                for stage in report['stages'].values():
                    assert stage['elements']==2*4*7*11
                    if mode=='zero':
                        assert stage['correction_rms']==0.
                    else:
                        assert stage['correction_rms']>0.
            torch.testing.assert_close(module(x),original,rtol=0,atol=0)
        # Check that padding cells are excluded from accumulated element counts.
        batch=dict(img=torch.zeros(2,3,28,44),ori_shape=[(20,36)]*2,
                   ratio_pad=[((1.,1.),(4,4))]*2)
        with GainIntervention(module) as observer:
            observer.set_batch(batch)
            module(x)
            for stage in observer.report()['stages'].values():
                assert stage['elements']==2*4*5*9
        try:
            with GainIntervention(module,'zero'):
                raise RuntimeError('exercise exception cleanup')
        except RuntimeError:
            pass
        torch.testing.assert_close(module(x),original,rtol=0,atol=0)
    for k,v in module.state_dict().items():
        torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    assert not any(m._forward_hooks for m in module.modules())
    return dict(learned_output_unchanged=True,three_interventions_match_manual=True,
                output_restored=True,exception_cleanup=True,state_dict_unchanged=True,
                nonzero_geometry_exercised=True,padding_excluded=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'status.json').exists():
        raise FileExistsError('Diagnostic status already exists; refusing duplicate run')
    torch.set_num_threads(4)
    source = json.loads(SOURCE_RESULT.read_text(encoding='utf-8'))
    checkpoint = Path(source['checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==source['checkpoint_sha256']
    state = dict(status='running',pid=os.getpid(),started_at=now(),updated_at=now(),
                 modes=list(MODES),completed=[],checkpoint=str(checkpoint),checkpoint_sha256=digest,
                 training_started=False)
    write_json(OUT/'status.json',state)
    checks = self_check()
    write_json(OUT/'self_checks.json',checks)
    print('SELF_CHECKS '+json.dumps(checks),flush=True)
    sources = ['diagnose_samx_geometry.py','diagnose_cfsav1_regions.py',
               'vajra/nn/samx_geometry.py','vajra/nn/modules.py','vajra/core/validator.py',
               'vajra/models/vajra/segment/val.py','vajra/nn/backend.py']
    write_json(OUT/'source_manifest.json',
               {k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in sources})
    records = {}
    started = time.monotonic()
    for mode in MODES:
        init_seeds(42,deterministic=True)
        state.update(current_mode=mode,updated_at=now(),images_processed=0)
        write_json(OUT/'status.json',state)
        args = {**source['validation_options'],'task':'segment','mode':'val','model':str(checkpoint),
                'project':str(OUT/'validation'),'name':mode,'exist_ok':False,
                'seed':42,'deterministic':True}
        validator = GeometryValidator(args=args,intervention=mode)
        stamp = [0.]
        def progress(v):
            current = time.monotonic()
            if current-stamp[0]>=5:
                stamp[0]=current
                state.update(updated_at=now(),images_processed=int(v.seen),batch=int(v.batch_i)+1,
                             elapsed_seconds=current-started)
                write_json(OUT/'status.json',state)
        validator.add_callback('on_val_batch_end',progress)
        print('START '+mode,flush=True)
        stats = validator(model=str(checkpoint))
        counts = dict(images=int(validator.seen),instances=int(validator.nt_per_class.sum()))
        assert counts==source['validation_counts'], counts
        if mode=='learned':
            for key,value in source['metrics'].items():
                assert abs(float(stats[key])-value)<1e-7, (key,stats[key],value)
        record = dict(mode=mode,fixed_gain=MODES[mode],finished_at=now(),
            checkpoint=str(checkpoint),checkpoint_sha256=digest,validation_options=args,
            counts=counts,metrics={k:float(v) for k,v in stats.items()},
            per_class={str(int(c)):dict(name=validator.names[int(c)],instances=int(validator.nt_per_class[c]),
                **dict(zip(('precision','recall','map50','map95'),
                           map(float,validator.metrics.seg.class_aware_results(i)))))
                for i,c in enumerate(validator.metrics.seg.ap_class_idx)},
            responses=validator.observer.report())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        record['checkpoint_unchanged']=True
        records[mode]=record
        write_json(OUT/(mode+'.json'),record)
        state['completed'].append(mode)
        state.update(images_processed=counts['images'],batch=len(validator.dataloader),updated_at=now())
        write_json(OUT/'status.json',state)
        print('DONE '+mode+' '+json.dumps(record),flush=True)
        del validator
        gc.collect()
        torch.cuda.empty_cache()
    baseline=records['learned']['metrics']
    deltas={mode:{k:100*(r['metrics'][k]-baseline[k]) for k in MASK_KEYS}
            for mode,r in records.items()}
    write_json(OUT/'comparison.json',dict(records=records,deltas_percentage_points=deltas,
        checkpoint_unchanged=True,baseline_reproduced=True,
        note='Same trained checkpoint, inference intervention only. No new architecture training; changing both stages does not isolate either stage. Validation-set tuning is not an independent test result.'))
    state.update(status='complete',current_mode=None,updated_at=now(),elapsed_seconds=time.monotonic()-started,
                 validation_counts_per_mode={k:r['counts'] for k,r in records.items()})
    write_json(OUT/'status.json',state)
    print('ALL_GEOMETRY_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        run()
    except BaseException:
        failure=dict(status='failed',updated_at=now(),traceback=traceback.format_exc())
        write_json(OUT/'failure.json',failure)
        if (OUT/'status.json').exists():
            state=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
            state.update(failure)
            write_json(OUT/'status.json',state)
        raise
