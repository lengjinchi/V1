"""Separate SAMX routing ablations; the original module stays in modules.py."""
import torch
from torch import nn

from vajra.nn.modules import ShapeAdaptiveMerudandaX


SAMX_MODES = ("off", "original", "stage_global", "local_global", "geometry_residual", "geometry_dual",
              "branch_evidence")


class StageGlobalSAMX(ShapeAdaptiveMerudandaX):
    """Experiment 2: separate global gates, each reading its own stage input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.in_c // self.route_reduction, 8)
        self.route = nn.Identity()  # no unused original gate in the optimizer
        self.route2 = self._make_gate(self.hidden_c, hidden)
        self.route3 = self._make_gate(self.mid_c2, hidden)

    @staticmethod
    def _make_gate(channels, hidden):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                             nn.SiLU(), nn.Conv2d(hidden, 2, 1))

    def forward(self, x):
        f1, f2 = self.conv1(x).chunk(2, 1)
        a2 = self.route2(f2).softmax(1)
        y2 = self.block2(f2, a2[:, :1], a2[:, 1:])
        a3 = self.route3(y2).softmax(1)
        y3 = self.block3(y2, a3[:, :1], a3[:, 1:])
        return self.conv4(torch.cat((f1, f2, y2, y3), 1))

    def forward_split(self, x):
        return self.forward(x)

    def get_module_info(self):
        return "StageGlobalSAMX", f"[{self.in_c}, {self.out_c}, stage-specific GAP gates]"


class LocalGlobalSAMX(ShapeAdaptiveMerudandaX):
    """Experiment 3: original shared global gate plus a shared local logit map.

    This is a separate ablation, not experiment 2 plus another modification.
    The local final layer starts at zero, recovering the original global gate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.in_c // self.route_reduction, 8)
        self.local_route = nn.Sequential(
            nn.Conv2d(self.in_c, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.zeros_(self.local_route[-1].weight)
        nn.init.zeros_(self.local_route[-1].bias)

    def _get_route_weight(self, x):
        alpha = (self.route(x) + self.local_route(x)).softmax(1)
        return alpha[:, :1], alpha[:, 1:]

    def get_module_info(self):
        return "LocalGlobalSAMX", f"[{self.in_c}, {self.out_c}, shared local+global gate]"


def configure_samx(backbone, mode, geometry_gain=0.1):
    """Replace only nano neck3; fail rather than silently run the wrong model."""
    if mode not in SAMX_MODES:
        raise ValueError(f"samx_mode must be one of {SAMX_MODES}, got {mode!r}")
    if mode == "off":
        return
    old = getattr(backbone, "vajra_neck3", None)
    expected = (192, 128, 128, 64)
    actual = tuple(getattr(old, key, None) for key in ("in_c", "out_c", "mid_c1", "mid_c2"))
    if actual != expected:
        raise ValueError(f"SAMX ablations require nano neck3 {expected}, got {actual}")
    from vajra.nn.samx_geometry import GeometryResidualSAMX
    from vajra.nn.samx_followups import DualPathGeometrySAMX
    from vajra.nn.samx_branch_evidence import BranchEvidenceSAMX
    cls = {"original": ShapeAdaptiveMerudandaX, "stage_global": StageGlobalSAMX,
           "local_global": LocalGlobalSAMX, "geometry_residual": GeometryResidualSAMX,
           "geometry_dual": DualPathGeometrySAMX, "branch_evidence": BranchEvidenceSAMX}[mode]
    # Extra gate initialization must not alter the later augmentation RNG stream.
    with torch.random.fork_rng(devices=[]):
        extra = {"geometry_gain": geometry_gain} if mode in ("geometry_residual","geometry_dual") else {}
        module = cls(*expected, num_blocks=1, rep_csp=True, ldconv_num_param=3, route_reduction=4, **extra)
    from vajra.utils.torch_utils import initialize_weights
    initialize_weights(module)
    # Preserve downstream head initialization/RNG state. The geometry variant
    # intentionally changes the internal feature paths to a shared trunk.
    parameter = next(old.parameters())
    backbone.vajra_neck3 = module.to(device=parameter.device, dtype=parameter.dtype)
