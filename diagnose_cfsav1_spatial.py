"""Same-checkpoint validation: dynamic gate maps vs their per-image spatial means.

Does not change training code or checkpoint files. Run from the project root:
    python diagnose_cfsav1_spatial.py
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from vajra.models.vajra.segment.val import SegmentationValidator
from vajra.nn.cfsav1 import CFSAv1
from vajra.utils.torch_utils import init_seeds

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = ROOT / ('CFSAV1 compare/'
    'VajraV1_CFSAV1_rho0.25_global1_smooth0_ramp0_start1.0_paired0_sens0_seed42/'
    'weights/best-vajra-v1-nano-seg.pt')


def captured_gate_logits(cache):
    """Recreate actual fused logits, including optional coarse P4 hints."""
    logits = cache['local']
    if 'global' in cache:
        logits = logits + cache['global']
    if 'semantic' in cache:
        logits = logits + F.interpolate(cache['semantic'], size=logits.shape[-2:],
                                        mode='bilinear', align_corners=False)
    return logits


class GateMapIntervention:
    """Observe actual post-tanh coefficients and optionally remove spatial variation.

    The original module runs unchanged. Hooks retain its bands/local/global
    logits only until its forward ends, then reconstruct output with mean gains.
    No averaging over images, groups or frequency bands is performed.
    """
    def __init__(self, model, spatial_mean=False):
        self.model, self.spatial_mean = model, spatial_mean
        self.handles, self.cache, self.modules = [], {}, []
        self.n = 0
        self.sums = {k: torch.zeros(3, dtype=torch.float64) for k in ('gain', 'spatial_std', 'removed_rms')}

    def __enter__(self):
        try:
            for m in self.model.modules():
                if not isinstance(m, CFSAv1):
                    continue
                if getattr(m, 'use_sensitivity', False):
                    raise ValueError('This experiment is for single-view CFSAv1, without legacy sensitivity heads.')
                key = id(m)
                self.modules.append({'groups': m.groups, 'rho': m.rho,
                    'gate_scale': float(getattr(m, 'gate_scale', 1.0)),
                    'smooth': bool(getattr(m, 'gate_smooth', False)),
                    'global_context': getattr(m, 'global_gate', None) is not None,
                    'semantic_context': getattr(m, 'semantic_gate', None) is not None})
                self.handles.append(m.register_forward_pre_hook(self._begin(key)))
                for name, sub in [('bands', m.bands), ('local', m.gate),
                                  ('global', getattr(m, 'global_gate', None)),
                                  ('semantic', getattr(m, 'semantic_gate', None))]:
                    if sub is not None:
                        self.handles.append(sub.register_forward_hook(self._capture(key, name)))
                self.handles.append(m.register_forward_hook(self._finish(key)))
            if not self.modules:
                raise ValueError('Checkpoint contains no CFSAv1 module.')
            return self
        except Exception:
            self.close()
            raise

    def _begin(self, key):
        def hook(module, args):
            self.cache[key] = {}
        return hook

    def _capture(self, key, name):
        def hook(module, args, output):
            self.cache[key][name] = output.detach()
        return hook

    def _finish(self, key):
        def hook(module, args, output):
            cache = self.cache.pop(key)
            logits = captured_gate_logits(cache)
            delta = module.rho * logits.tanh() * getattr(module, 'gate_scale', 1.0)
            if getattr(module, 'gate_smooth', False):
                delta = F.avg_pool2d(delta, 3, 1, 1, count_include_pad=False)
            # Average AFTER nonlinear bounding and optional coefficient smoothing.
            # mean(tanh(z)) != tanh(mean(z)). Global context is already included.
            mean = delta.mean((-2, -1), keepdim=True)
            b, _, h, w = delta.shape
            d = delta.float().reshape(b, module.groups, 3, h*w)
            spatial_std = d.std(-1, unbiased=False)
            self.sums['gain'] += (1+d.mean(-1)).mean(1).sum(0).double().cpu()
            self.sums['spatial_std'] += spatial_std.mean(1).sum(0).double().cpu()
            self.sums['removed_rms'] += ((d-d.mean(-1, keepdim=True)).square().mean(-1).sqrt()).mean(1).sum(0).double().cpu()
            self.n += b
            if not self.spatial_mean:
                return output
            x = args[0]
            b, c, h, w = x.shape
            bands = cache['bands'].reshape(b, module.groups, c // module.groups, 3, h, w)
            mean = mean.reshape(b, module.groups, 1, 3, 1, 1)
            return x + (mean * bands).sum(3).reshape_as(x).to(x.dtype)
        return hook

    def report(self):
        return {'mode': 'spatial_mean' if self.spatial_mean else 'dynamic',
                'image_module_evaluations': self.n, 'modules': self.modules,
                'band_order': ['low', 'middle', 'high'],
                **{k: (v / max(self.n, 1)).tolist() for k, v in self.sums.items()}}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.cache.clear()

    def __exit__(self, *exc):
        self.close()


class DiagnosticValidator(SegmentationValidator):
    def __init__(self, *args, spatial_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_mean = spatial_mean
        self.intervention = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.intervention = GateMapIntervention(model, self.spatial_mean)
        self.intervention.__enter__()

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if self.intervention is not None:
                self.intervention.close()


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def run(args):
    weight = args.weights.resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    args.output.mkdir(parents=True, exist_ok=True)
    before_hash = digest(weight)
    records = {}
    common = dict(task='segment', mode='val', model=str(weight), data=str(args.data.resolve()),
        split='val', img_size=1600, rect=True, batch=args.batch, workers=0,
        device=args.device, half=False, conf=.001, iou=.7, max_det=300,
        overlap_mask=True, mask_ratio=4, augment=False, plots=False,
        project=str(args.output.resolve()), exist_ok=True, seed=42, deterministic=True,
        save_json=False, save_txt=False, verbose=False)
    for mode in ['dynamic', 'spatial_mean']:
        init_seeds(42, deterministic=True)
        validator = DiagnosticValidator(args={**common, 'name': mode}, spatial_mean=mode=='spatial_mean')
        stats = validator(model=str(weight))
        per_class = {str(int(c)): dict(zip(['precision', 'recall', 'map50', 'map95'],
                     map(float, validator.metrics.seg.class_aware_results(i))))
                     for i, c in enumerate(validator.metrics.seg.ap_class_idx)}
        records[mode] = {'metrics': {k:float(v) for k,v in stats.items()},
            'per_class': per_class, 'names': validator.names,
            'gate_statistics': validator.intervention.report(), 'images': validator.seen}
        (args.output/f'{mode}.json').write_text(json.dumps(records[mode], indent=2, ensure_ascii=False), encoding='utf-8')
        del validator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert digest(weight)==before_hash, 'Checkpoint file changed during diagnostic.'
    delta = {k:100*(v-records['dynamic']['metrics'][k]) for k,v in records['spatial_mean']['metrics'].items()
             if k.startswith('metrics/')}
    report = {'checkpoint': str(weight), 'checkpoint_sha256': before_hash, 'settings': common,
              'comparison': records, 'spatial_mean_minus_dynamic_percentage_points': delta,
              'interpretation': 'Inference-only intervention; measures reliance on spatial gate variation of this checkpoint, not whether locally conditioned training is unnecessary. Timing is not a speed benchmark.'}
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),encoding='utf-8')
    lines = ['# CFSAv1 spatial gate intervention', '', '| Mask metric | Dynamic | Spatial mean | Difference (pp) |',
             '|---|---:|---:|---:|']
    for key in ['precision', 'recall', 'mAP50', 'mAP50-95']:
        k=f'metrics/{key}(Mask)'
        lines.append(f"| {key} | {100*records['dynamic']['metrics'][k]:.4f}% | {100*records['spatial_mean']['metrics'][k]:.4f}% | {delta[k]:+.4f} |")
    lines += ['', 'Same weights, full validation split, FP32. Each group/band map is averaged independently for each image.',
              'Global-context effects and per-image mean gains are retained. No checkpoint or training code is modified.',
              'P/R follow each validator-selected operating point. Use AP and per-class results as primary evidence.',
              'Differences diagnose this checkpoint only. No architectural or statistical significance claim follows automatically.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument('--data', type=Path, default=Path(r'F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'output/cfsav1_global_spatial_diagnostic')
    run(parser.parse_args())
