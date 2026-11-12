"""Opt-in SAMX routing conditioned on each stage's actual branch outputs."""
import torch
from torch import nn

from vajra.nn.samx import LocalGlobalSAMX


class BranchEvidenceSAMX(LocalGlobalSAMX):
    """Keep both paths and the old gate; add one shared six-map correction head.

    The correction head reads each stage separately and is called twice, with
    shared parameters. Its zero-initialized last layer recovers LocalGlobal at
    construction. No feature tensors or computation graphs are retained as state.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization must not change the common initialization
        # or subsequent augmentation RNG stream, including direct construction.
        with torch.random.fork_rng(devices=[]):
            self.evidence_route = nn.Sequential(
                nn.Conv2d(6, 8, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, 2, 1),
            )
        nn.init.zeros_(self.evidence_route[-1].weight)
        nn.init.zeros_(self.evidence_route[-1].bias)

    @staticmethod
    def branch_statistics(regular, deform):
        # Reduce in FP32 for AMP stability, then honor explicit half inference.
        r, d = regular.float(), deform.float()
        difference = (r - d).abs()
        return torch.cat([
            value
            for feature in (r, d, difference)
            for value in (feature.mean(1, keepdim=True), feature.amax(1, keepdim=True))
        ], 1).to(regular.dtype)

    def branch_weights(self, base_logits, regular, deform):
        correction = self.evidence_route(self.branch_statistics(regular, deform))
        return (base_logits + correction).softmax(1)

    def _fuse_stage(self, block, x, base_logits):
        regular = block.regular_path(x)
        deform = block.deform_path(x)
        alpha = self.branch_weights(base_logits, regular, deform)
        return alpha[:, :1] * regular + alpha[:, 1:] * deform

    def forward(self, x):
        f1, f2 = self.conv1(x).chunk(2, 1)
        base_logits = self.route(x) + self.local_route(x)
        y2 = self._fuse_stage(self.block2, f2, base_logits)
        y3 = self._fuse_stage(self.block3, y2, base_logits)
        return self.conv4(torch.cat((f1, f2, y2, y3), 1))

    def forward_split(self, x):
        return self.forward(x)

    def get_module_info(self):
        return "BranchEvidenceSAMX", f"[{self.in_c}, {self.out_c}, shared branch-output gate correction]"
