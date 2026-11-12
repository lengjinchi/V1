"""Observe CFSAv1 gates in ground-truth foreground, boundary and background.

Inference only: labels select reporting regions, never model inputs or outputs.
Run from the project root: python diagnose_cfsav1_regions.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_cfsav1_spatial import DEFAULT_WEIGHT, ROOT, GateMapIntervention, captured_gate_logits, digest
from vajra.models.vajra.segment.val import SegmentationValidator
from vajra.utils.torch_utils import init_seeds


REGIONS = ('foreground', 'interior', 'boundary', 'background', 'near_background', 'far_background')
CONTRASTS = (('foreground', 'background'), ('interior', 'far_background'),
             ('boundary', 'far_background'), ('near_background', 'far_background'))


def valid_mask(batch, mask_shape):
    """Keep only mask cells fully inside the resized, unpadded image rectangle."""
    ih, iw = batch['img'].shape[-2:]
    mh, mw = mask_shape
    device = batch['img'].device
    ys = torch.arange(mh, device=device) * (ih / mh)
    xs = torch.arange(mw, device=device) * (iw / mw)
    result = []
    for original, ratio_pad in zip(batch['ori_shape'], batch['ratio_pad']):
        (gy, gx), (left, top) = ratio_pad
        right = left + round(original[1] * gx)
        bottom = top + round(original[0] * gy)
        if not (0 <= left < right <= iw and 0 <= top < bottom <= ih):
            raise ValueError(f'Invalid image rectangle: {original}, {ratio_pad}, {(ih, iw)}')
        vy = (ys >= top) & (ys + ih / mh <= bottom + 1e-5)
        vx = (xs >= left) & (xs + iw / mw <= right + 1e-5)
        result.append(vy[:, None] & vx[None, :])
    return torch.stack(result)[:, None]


def make_regions(foreground, valid, radius, all_foreground=None):
    """Morphology on native GT masks, before area projection to the gate grid.

    Boundary is the inner boundary of the foreground union; outside pixels are
    reported separately as near background. All labelled classes are excluded
    from class-specific background comparisons. Padding is always excluded.
    """
    foreground = foreground.bool() & valid
    all_foreground = foreground if all_foreground is None else all_foreground.bool() & valid
    k = 2 * radius + 1
    padded = F.pad(foreground.float(), (radius,) * 4, value=0)
    interior = -F.max_pool2d(-padded, k, 1) > .5
    dilated = F.max_pool2d(foreground.float(), k, 1, radius) > 0
    all_dilated = F.max_pool2d(all_foreground.float(), k, 1, radius) > 0
    background = valid & ~all_foreground
    regions = (foreground, foreground & interior, foreground & ~interior,
               background, background & dilated, background & ~all_dilated)
    return torch.cat(regions, 1)


def weighted_region_means(values, regions):
    """Area coverage preserves thin masks; no nearest-neighbour label sampling."""
    weights = F.interpolate(regions.float(), size=values.shape[-2:], mode='area')
    mass = weights.sum((-2, -1))
    means = torch.einsum('bdhw,brhw->brd', values, weights) / mass.clamp_min(1e-12)[..., None]
    means = means.masked_fill(mass[..., None] == 0, torch.nan)
    return means, mass


class RegionGateObserver(GateMapIntervention):
    """Retain actual gains and grouped band responses; return output unchanged."""
    def __init__(self, model):
        super().__init__(model, spatial_mean=False)
        self.latest = None

    def _finish(self, key):
        parent_hook = super()._finish(key)

        def hook(module, args, output):
            if self.latest is not None:
                raise RuntimeError('Unconsumed gate maps, or multiple CFSAv1 modules.')
            cache = self.cache[key]
            logits = captured_gate_logits(cache)
            delta = module.rho * logits.tanh() * getattr(module, 'gate_scale', 1.0)
            if getattr(module, 'gate_smooth', False):
                delta = F.avg_pool2d(delta, 3, 1, 1, count_include_pad=False)
            b, c, _, h, w = cache['bands'].shape
            response = cache['bands'].float().reshape(b, module.groups, c // module.groups, 3, h, w).abs().mean(2)
            fractions = response / response.sum(2, keepdim=True).clamp_min(1e-12)
            # Keep each group's gains separately to detect cancellation in group means.
            self.latest = torch.cat((1 + delta.float(), response.mean(1), fractions.mean(1)), 1)
            return parent_hook(module, args, output)
        return hook

    def close(self):
        self.latest = None
        super().close()


def describe(values):
    """Statistics over images, retaining the group/band dimensions."""
    values = np.asarray(values, dtype=float)
    return {'mean': values.mean(0).tolist(), 'mean_absolute': np.abs(values).mean(0).tolist(),
            'p95_absolute': np.quantile(np.abs(values), .95, axis=0).tolist()}


class RegionCollector:
    def __init__(self, groups):
        self.groups = groups
        self.records = {}

    def add(self, scope, radius, files, means, mass, include=None):
        key = f'{scope}/radius{radius}'
        rows = self.records.setdefault(key, [])
        means, mass = means.detach().cpu().numpy(), mass.detach().cpu().numpy()
        include = np.ones(len(files), bool) if include is None else include.cpu().numpy().astype(bool)
        for i, file in enumerate(files):
            if include[i]:
                rows.append({'image': str(file), 'means': means[i], 'mass': mass[i]})

    def summary(self):
        result = {}
        for scope, rows in self.records.items():
            if not rows:
                continue
            means = np.stack([r['means'] for r in rows])
            mass = np.stack([r['mass'] for r in rows])
            regions, contrasts = {}, {}
            n_gains = self.groups * 3
            for i, name in enumerate(REGIONS):
                present = mass[:, i] > 0
                if not present.any():
                    regions[name] = {'images': 0}
                    continue
                v, w = means[present, i], mass[present, i]
                gain = v[:, :n_gains].reshape(-1, self.groups, 3)
                regions[name] = {
                    'images': int(present.sum()), 'equivalent_feature_cells': float(w.sum()),
                    'gain_image_mean_by_group': gain.mean(0).tolist(),
                    'gain_image_mean': gain.mean((0, 1)).tolist(),
                    'gain_area_weighted_mean': np.average(gain, axis=0, weights=w).mean(0).tolist(),
                    'absolute_response_image_mean': v[:, n_gains:n_gains+3].mean(0).tolist(),
                    'response_fraction_image_mean': v[:, n_gains+3:n_gains+6].mean(0).tolist(),
                }
            for first, second in CONTRASTS:
                a, b = REGIONS.index(first), REGIONS.index(second)
                present = (mass[:, a] > 0) & (mass[:, b] > 0)
                if not present.any():
                    contrasts[f'{first}_minus_{second}'] = {'paired_images': 0}
                    continue
                difference = (means[present, a, :n_gains] - means[present, b, :n_gains]).reshape(-1, self.groups, 3)
                contrasts[f'{first}_minus_{second}'] = {
                    'paired_images': int(present.sum()),
                    'by_group': describe(difference),
                    'signed_mean_by_band': difference.mean((0, 1)).tolist(),
                    'mean_absolute_by_band': np.abs(difference).mean((0, 1)).tolist(),
                    'p95_absolute_image_group_by_band': np.quantile(np.abs(difference).reshape(-1, 3), .95, axis=0).tolist(),
                }
            result[scope] = {'images': len(rows), 'regions': regions, 'paired_contrasts': contrasts}
        return result

    def write_rows(self, path):
        with path.open('w', encoding='utf-8') as stream:
            for scope, rows in self.records.items():
                for row in rows:
                    # Empty regions are JSON null, not invalid JSON NaN values.
                    means = row['means'].astype(object)
                    means[~np.isfinite(row['means'])] = None
                    stream.write(json.dumps({'scope': scope, 'image': row['image'],
                        'region_order': REGIONS, 'means': means.tolist(),
                        'equivalent_feature_cells': row['mass'].tolist()}, allow_nan=False) + '\n')


class RegionDiagnosticValidator(SegmentationValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = None
        self.collector = None
        self.geometry = set()

    def init_metrics(self, model):
        super().init_metrics(model)
        self.observer = RegionGateObserver(model)
        self.observer.__enter__()
        if len(self.observer.modules) != 1:
            raise ValueError('This diagnostic requires exactly one CFSAv1 module.')
        self.collector = RegionCollector(self.observer.modules[0]['groups'])

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        values, self.observer.latest = self.observer.latest, None
        if values is None:
            raise RuntimeError('No gate maps captured.')
        masks = batch['masks']
        if not self.args.overlap_mask or len(masks) != len(values):
            raise ValueError('This diagnostic uses one overlap-ID mask per image.')
        valid = valid_mask(batch, masks.shape[-2:])
        foreground = (masks[:, None] > 0) & valid
        # IDs follow the dataset's area-sorted cls order, including overlap ownership.
        label_map = torch.full_like(masks, -1, dtype=torch.long)
        for i in range(len(masks)):
            classes = batch['cls'][batch['batch_idx'] == i].flatten().long()
            lookup = torch.cat((classes.new_tensor([-1]), classes))
            ids = masks[i].long()
            if int(ids.max()) >= len(lookup):
                raise ValueError('Overlap mask ID does not match the annotation class order.')
            label_map[i] = lookup[ids]
        geometry = (*batch['img'].shape[-2:], *masks.shape[-2:], *values.shape[-2:])
        self.geometry.add(geometry)
        if geometry[0] != 4 * geometry[2] or geometry[1] != 4 * geometry[3]:
            raise ValueError('Expected mask_ratio=4 for the 8/16px boundary definitions.')
        for radius in (2, 4):
            regions = make_regions(foreground, valid, radius)
            means, mass = weighted_region_means(values, regions)
            self.collector.add('all', radius, batch['im_file'], means, mass)
        # Class-specific foreground; background always excludes every labelled defect.
        for cid in range(self.num_classes):
            fg_class = (label_map[:, None] == cid) & valid
            include = fg_class.flatten(1).any(1)
            if not include.any():
                continue
            regions = make_regions(fg_class, valid, 2, all_foreground=foreground)
            means, mass = weighted_region_means(values, regions)
            self.collector.add(f'class{cid}', 2, batch['im_file'], means, mass, include)

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if self.observer is not None:
                self.observer.close()


def save_report(path, result):
    stats = result['regions']['all/radius2']
    contrast = stats['paired_contrasts']['foreground_minus_background']
    differences = contrast.get('mean_absolute_by_band')
    if differences is None:
        findings = ['缺少同时含前景和背景的有效图像，无法计算缺陷与背景的成对增益差。']
    else:
        findings = [
            '缺陷与背景的同图平均绝对增益差（低/中/高频）为：' +
            ' / '.join(f'{x:.8f}' for x in differences) + '。',
            '相对于单位增益1，分别为：' + ' / '.join(f'{100*x:.6f}%' for x in differences) + '。',
        ]
    lines = ['# CFSAv1 缺陷区域与背景权重诊断', '',
        f"完整验证 {result['images']} 张图。使用同一最佳权重、FP32，仅观察门控，不修改模型输出。", '',
        '## 主要发现', '', *findings,
        '空间变化是否有助于分割，需要结合该权重自身的空间均值干预和同条件性能对照判断。',
        '各模型必须根据各自实测数值解释，不能沿用此前模型的结论。', '',
        '## 区域划分', '',
        '依据人工标注的可见缺陷并集划分：内部为腐蚀后区域；边界为缺陷内部被腐蚀去掉的区域；',
        '近背景位于缺陷外侧膨胀带；远背景位于膨胀带之外。主分析宽度约8个输入像素，另检查16像素。',
        '排除补边，仅使用完全位于真实图像内的标注网格单元。采用面积覆盖率映射到门控网格，避免细缺陷被最近邻下采样抹掉。',
        '内部为空的图像不参与内部对比，单独报告有效图像数。边界不是实例之间的分界。', '',
        '## 平均增益（1代表不改变该频带）', '',
        '| 区域 | 有效图像数 | 低频 | 中频 | 高频 |', '| --- | ---: | ---: | ---: | ---: |']
    region_names = {'foreground':'全部缺陷', 'interior':'缺陷内部', 'boundary':'缺陷内边界',
                    'background':'全部背景', 'near_background':'缺陷附近背景', 'far_background':'远离缺陷的背景'}
    for name, row in stats['regions'].items():
        if row['images']:
            lines.append(f"| {region_names[name]} | {row['images']} | " + ' | '.join(f'{x:.8f}' for x in row['gain_image_mean']) + ' |')
    lines += ['', '上表先在每张图中求区域均值，再对有效图像和通道组平均。不同区域有效图像可能不同，应以以下成对差值为主要依据。', '',
              '## 同图成对差异：各通道组差值绝对值的平均', '',
              '| 区域对比 | 有效图像数 | 低频 | 中频 | 高频 |', '| --- | ---: | ---: | ---: | ---: |']
    for (first, second), row in zip(CONTRASTS, stats['paired_contrasts'].values()):
        if row['paired_images']:
            lines.append(f"| {region_names[first]} vs {region_names[second]} | {row['paired_images']} | " +
                         ' | '.join(f'{x:.8f}' for x in row['mean_absolute_by_band']) + ' |')
    lines += ['', '差值保留各通道组后取绝对值，可避免组间正负抵消；有符号差值、各组结果和95%分位数见 regions.json。',
        '这里报告效应大小，不将极小非零差值直接解释为有用的缺陷识别能力。', '',
        '## 各类别：缺陷与背景的同图平均绝对增益差', '',
        '| 类别 | 有效图像数 | 低频 | 中频 | 高频 |', '| --- | ---: | ---: | ---: | ---: |']
    for cid, name in result['names'].items():
        scope = result['regions'].get(f'class{cid}/radius2')
        if scope is None:
            continue
        row = scope['paired_contrasts']['foreground_minus_background']
        lines.append(f"| {name} | {row['paired_images']} | " +
                     ' | '.join(f'{x:.8f}' for x in row['mean_absolute_by_band']) + ' |')
    lines += ['', '类别可在同一图像中共存，表内图像数不能相加作为总图像数。', '',
        '## 解释范围', '',
        '这是观察性诊断：区域权重不同不等于这种差异能提高分割性能；接近也不能证明门控输入完全没有信息。',
        '类别分析使用该类标注并集作为前景，并从背景中排除所有类别的缺陷。原始幅度响应和其占比仅描述特征，不能当成缺陷概率。',
        '16像素结果、各类别统计、逐图原始统计和图像/标注/门控网格尺寸均已保存。',
        '当前仅诊断该份最佳权重；没有重新训练或进行独立测试集验证。', '',
        f"权重文件SHA256校验通过：`{result['checkpoint_sha256']}`。"]
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def save_figure(path, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout='constrained')
    labels = ['Defect / background', 'Interior / far BG', 'Boundary / far BG', 'Near BG / far BG']
    colors = ['#3977B8', '#DE8F35', '#4E9D71']
    for ax, radius in zip(axes, (2, 4)):
        rows = summary[f'all/radius{radius}']['paired_contrasts']
        x = np.arange(len(labels))
        for band, name in enumerate(('Low', 'Middle', 'High')):
            y = [r.get('mean_absolute_by_band', [np.nan]*3)[band] for r in rows.values()]
            ax.bar(x + (band-1)*.24, y, .24, label=name, color=colors[band])
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.set_ylabel('Mean absolute gain difference')
        ax.set_title(f'Boundary width: approximately {4*radius} input pixels')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    weight = args.weights.resolve()
    before_hash = digest(weight)
    args.output.mkdir(parents=True, exist_ok=True)
    init_seeds(42, deterministic=True)
    settings = dict(task='segment', mode='val', model=str(weight), data=str(args.data.resolve()),
        split='val', img_size=1600, rect=True, batch=args.batch, workers=0, device=args.device,
        half=False, conf=.001, iou=.7, max_det=300, overlap_mask=True, mask_ratio=4,
        augment=False, plots=False, project=str(args.output.resolve()), name='observed', exist_ok=True,
        seed=42, deterministic=True, save_json=False, save_txt=False, verbose=False)
    validator = RegionDiagnosticValidator(args=settings)
    metrics = validator(model=str(weight))
    assert digest(weight) == before_hash, 'Checkpoint file changed.'
    assert validator.seen == len(validator.dataloader.dataset), 'Incomplete validation.'
    assert len(validator.collector.records['all/radius2']) == validator.seen
    result = {'checkpoint': str(weight), 'checkpoint_sha256': before_hash, 'settings': settings,
        'images': validator.seen, 'instances': int(validator.nt_per_class.sum()), 'names': validator.names,
        'geometry_order': ['input_h','input_w','gt_h','gt_w','gate_h','gate_w'],
        'geometries': sorted(validator.geometry), 'group_count': validator.collector.groups,
        'raw_value_order': 'group-major low/middle/high gains, 3 absolute responses, 3 response fractions',
        'metrics': {k:float(v) for k,v in metrics.items()}, 'gates': validator.observer.report(),
        'regions': validator.collector.summary()}
    previous = ROOT/'output/cfsav1_global_spatial_diagnostic/dynamic.json'
    if previous.is_file():
        prior = json.loads(previous.read_text(encoding='utf-8'))
        result['previous_dynamic_metric_difference'] = {k:float(v-prior['metrics'][k]) for k,v in metrics.items()}
    (args.output/'regions.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    validator.collector.write_rows(args.output/'per_image.jsonl')
    save_report(args.output/'report.md', result)
    save_figure(args.output/'region_contrasts.png', result['regions'])
    print((args.output/'report.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument('--data', type=Path, default=Path(r'F:\material\VayuAI-mainV1\VayuAI-main\datasets\data.yaml'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'output/cfsav1_region_diagnostic')
    run(parser.parse_args())
