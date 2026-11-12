"""Inference-only SAMX mechanism checks. Never trains or writes checkpoints."""
import gc
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch

from diagnose_cfsav1_regions import valid_mask
from vajra.models.vajra.segment.val import SegmentationValidator
from vajra.nn.modules import LDConv, ShapeAdaptiveMerudandaX
from vajra.nn.samx import LocalGlobalSAMX
from vajra.utils.torch_utils import init_seeds

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/samx_diagnostic_20261028'
AUDIT = ROOT / 'output/samx_experiments_20261027'


class SAMXObserver:
    def __init__(self, model, mode='dynamic'):
        self.model, self.mode = model, mode
        self.handles, self.restore = [], []
        self.gates, self.offsets, self.branches = {}, {}, {}
        self.batch = None

    def __enter__(self):
        for index, module in enumerate(m for m in self.model.modules()
                                       if isinstance(m, ShapeAdaptiveMerudandaX)):
            if self.mode in ('spatial_mean', 'global_only'):
                if not isinstance(module, LocalGlobalSAMX):
                    raise ValueError('Local-gate interventions require LocalGlobalSAMX')
                previous = module.__dict__.get('_get_route_weight')
                original = module._get_route_weight
                def route(this, x, original=original):
                    if self.mode == 'global_only':
                        alpha = this.route(x).softmax(1)
                        return alpha[:, :1], alpha[:, 1:]
                    ac, al = original(x)
                    # Per-image mean of actual post-softmax coefficients.
                    # Padding is included in the intervention to preserve the
                    # full-map mean; descriptive statistics below exclude it.
                    return ac.mean((-2,-1),keepdim=True), al.mean((-2,-1),keepdim=True)
                module._get_route_weight = MethodType(route, module)
                self.restore.append((module, previous))
            for stage in ('block2','block3'):
                block = getattr(module,stage)
                key = f'{index}.{stage}'
                self.handles.append(block.register_forward_pre_hook(self.gate_hook(key)))
                for branch in ('regular_path','deform_path'):
                    self.handles.append(getattr(block,branch).register_forward_hook(self.branch_hook(key+'.'+branch)))
                for sub in block.modules():
                    if isinstance(sub,LDConv):
                        self.handles.append(sub.p_conv.register_forward_hook(self.offset_hook(key,sub)))
        if not self.handles:
            raise ValueError('No SAMX found')
        return self

    def mask(self, h, w, tensor):
        if self.batch is None:
            return torch.ones(tensor.shape[0],1,h,w,device=tensor.device,dtype=torch.bool)
        return valid_mask(self.batch,(h,w))

    def gate_hook(self,key):
        def hook(module,args):
            alpha=args[2].detach().float()
            h,w=alpha.shape[-2:]
            v=self.mask(h,w,alpha) if (h,w)!=(1,1) else torch.ones_like(alpha,dtype=torch.bool)
            sums=self.gates.setdefault(key,dict(images=0,ld_weight_sum=0.,spatial_std_sum=0.,saturated_fraction_sum=0.))
            for a,valid in zip(alpha,v):
                values=a[valid]
                if not values.numel():continue
                sums['images']+=1
                sums['ld_weight_sum']+=values.mean().item()
                sums['spatial_std_sum']+=values.std(unbiased=False).item()
                sums['saturated_fraction_sum']+=((values<.05)|(values>.95)).float().mean().item()
        return hook

    def branch_hook(self,key):
        def hook(module,args,output):
            x=output.detach().float()
            v=self.mask(*x.shape[-2:],x)
            s=self.branches.setdefault(key,dict(squared_sum=0.,elements=0))
            s['squared_sum']+=(x.square()*v).sum().item()
            s['elements']+=int(v.sum())*x.shape[1]
        return hook

    def offset_hook(self,key,ld):
        def hook(module,args,output):
            raw=output.detach().float()
            effective=torch.zeros_like(raw) if self.mode=='zero_offset' else raw
            b,_,h,w=raw.shape;n=ld.num_param
            valid=self.mask(h,w,raw).expand(b,n,h,w)
            norms=(effective[:,:n].square()+effective[:,n:].square()).sqrt()[valid]
            yy,xx=torch.meshgrid(torch.arange(h,device=raw.device)*ld.stride,
                                 torch.arange(w,device=raw.device)*ld.stride,indexing='ij')
            py=yy[None,None]+ld.p_n[:,:n]+effective[:,:n]
            px=xx[None,None]+ld.p_n[:,n:]+effective[:,n:]
            clipped=((py<0)|(py>args[0].shape[-2]-1)|(px<0)|(px>args[0].shape[-1]-1))[valid]
            s=self.offsets.setdefault(key,dict(points=0,norm_sum=0.,norm_squared_sum=0.,clipped_points=0,
                                                  above={str(t):0 for t in (1,2,4,8,16)}))
            s['points']+=norms.numel();s['norm_sum']+=norms.sum().item()
            s['norm_squared_sum']+=norms.square().sum().item();s['clipped_points']+=clipped.sum().item()
            for t in s['above']:s['above'][t]+=(norms>float(t)).sum().item()
            if self.mode=='zero_offset':return effective.to(output.dtype)
        return hook

    def report(self):
        return dict(mode=self.mode,
            gates={k:dict(images=s['images'],ld_weight_mean=s['ld_weight_sum']/s['images'],
                          spatial_std=s['spatial_std_sum']/s['images'],
                          saturation_fraction=s['saturated_fraction_sum']/s['images']) for k,s in self.gates.items()},
            offsets={k:dict(points=s['points'],mean_norm_feature_pixels=s['norm_sum']/s['points'],
                            rms_norm_feature_pixels=(s['norm_squared_sum']/s['points'])**.5,
                            feature_map_clipping_fraction=s['clipped_points']/s['points'],
                            fraction_above={t:v/s['points'] for t,v in s['above'].items()}) for k,s in self.offsets.items()},
            branch_rms={k:(s['squared_sum']/s['elements'])**.5 for k,s in self.branches.items()},
            note='Descriptive statistics exclude padding anchor cells. Clipping concerns feature-map bounds, not GT boundaries. Inference interventions are not retrained architectures.')

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        for module,previous in self.restore:
            if previous is None:del module._get_route_weight
            else:module._get_route_weight=previous
        self.restore=[]

    def __exit__(self,*args):self.close()


class DiagnosticValidator(SegmentationValidator):
    def __init__(self,*args,intervention='dynamic',**kwargs):
        super().__init__(*args,**kwargs)
        self.intervention=intervention;self.observer=None

    def init_metrics(self,model):
        super().init_metrics(model)
        self.observer=SAMXObserver(model,self.intervention).__enter__()

    def preprocess(self,batch):
        batch=super().preprocess(batch)
        self.observer.batch=batch
        return batch

    def __call__(self,*args,**kwargs):
        try:return super().__call__(*args,**kwargs)
        finally:
            if self.observer:self.observer.close()


def self_check():
    torch.manual_seed(42)
    module=LocalGlobalSAMX(8,8,8,4,rep_csp=True).eval()
    torch.nn.init.normal_(module.local_route[-1].weight,std=.2)
    x=torch.randn(2,8,5,9)
    with torch.inference_mode():
        expected=module(x)
        with SAMXObserver(module) as observer:torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        with SAMXObserver(module,'spatial_mean'):
            ac,al=module._get_route_weight(x)
            assert ac.shape[-2:]==(1,1)
            torch.testing.assert_close(ac+al,torch.ones_like(ac))
        with SAMXObserver(module,'global_only'):
            ac,al=module._get_route_weight(x)
            torch.testing.assert_close(al,module.route(x).softmax(1)[:,1:])
        with SAMXObserver(module,'zero_offset') as observer:
            module(x)
            assert all(s['norm_sum']==0 for s in observer.offsets.values())
        torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        assert '_get_route_weight' not in module.__dict__
    print('Observer/intervention self-check passed',flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    self_check()
    records={}
    for experiment,modes in [('stage_global',['dynamic']),('local_global',['dynamic','spatial_mean','global_only','zero_offset'])]:
        result=json.loads((AUDIT/(experiment+'_result.json')).read_text(encoding='utf-8'))
        weight=Path(result['checkpoint']);digest=hashlib.sha256(weight.read_bytes()).hexdigest()
        assert digest==result['checkpoint_sha256']
        for mode in modes:
            label=experiment+'_'+mode
            args={**result['validation_options'],'task':'segment','mode':'val','model':str(weight),
                  'project':str(OUT),'name':label,'exist_ok':True,'seed':42,'deterministic':True}
            init_seeds(42,deterministic=True)
            print('START',label,flush=True)
            validator=DiagnosticValidator(args=args,intervention=mode)
            stats=validator(model=str(weight))
            assert validator.seen==1532 and int(validator.nt_per_class.sum())==3942
            if mode=='dynamic':
                for key,value in result['metrics'].items():
                    assert abs(float(stats[key])-value)<1e-7,(label,key,value,stats[key])
            record=dict(experiment=experiment,intervention=mode,metrics={k:float(v) for k,v in stats.items()},
                        per_class={str(int(c)):dict(zip(['precision','recall','map50','map95'],
                              map(float,validator.metrics.seg.class_aware_results(i))))
                              for i,c in enumerate(validator.metrics.seg.ap_class_idx)},
                        names=validator.names,statistics=validator.observer.report(),
                        images=validator.seen,instances=int(validator.nt_per_class.sum()),checkpoint_sha256=digest)
            records[label]=record
            (OUT/(label+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print('DONE',label,json.dumps(record,ensure_ascii=False),flush=True)
            del validator;gc.collect();torch.cuda.empty_cache()
        assert hashlib.sha256(weight.read_bytes()).hexdigest()==digest
    (OUT/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL DIAGNOSTICS COMPLETED',flush=True)


if __name__=='__main__':run()
