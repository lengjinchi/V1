"""Geometric difference residuals at the original SAMX P4 insertion point."""
import torch
from torch import nn
from torch.nn import functional as F

from vajra.nn.modules import (ConvBNAct, LDConv, RepCSP, ShapeAdaptiveMerudandaX,
                              VajraV2InnerBlock)


class ResidualScale(nn.Module):
    """A scalar stored as `weight` so the framework optimizer includes it."""
    def __init__(self, initial=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(initial)))

    def forward(self, x):
        return x * self.weight.to(x.dtype)


class GeometricDifference(LDConv):
    """K[S(T,q+offset)-S(T,q)], with one shared bias-free linear K.

    Subtract fixed samples before weighted interpolation. This avoids
    cancellation of large branch outputs and exactly cancels constant inputs,
    including in half precision. No BN/activation runs inside either sample arm.
    LDConv inheritance keeps the established offset optimizer grouping intact.
    """
    def __init__(self, channels, num_param=3):
        super().__init__(channels, channels, num_param, stride=1)
        self.conv = nn.Identity()  # discard LDConv's aggregate BN and activation
        self.aggregate = nn.Conv2d(channels, channels, (num_param, 1),
                                   stride=(num_param, 1), bias=False)
        positions = self.p_n.reshape(2, num_param).tolist()
        self.fixed_positions = tuple(zip(positions[0], positions[1]))

    def fixed_samples(self, x):
        h, w = x.shape[-2:]
        max_y = max(y for y, _ in self.fixed_positions)
        max_x = max(xp for _, xp in self.fixed_positions)
        padded = F.pad(x, (0, max_x, 0, max_y), mode="replicate")
        # Integer reference sampling uses slices, not a second interpolator.
        return torch.stack([padded[:, :, y:y+h, xp:xp+w]
                            for y, xp in self.fixed_positions], dim=-1)

    def forward(self, x):
        offset = self.p_conv(x).float()
        n = self.num_param
        p = self._get_p(offset, offset.type()).permute(0, 2, 3, 1)
        py = torch.nan_to_num(p[..., :n], nan=0., posinf=x.shape[-2]-1, neginf=0.).clamp(0, x.shape[-2]-1)
        px = torch.nan_to_num(p[..., n:], nan=0., posinf=x.shape[-1]-1, neginf=0.).clamp(0, x.shape[-1]-1)
        y0, x0 = py.detach().floor(), px.detach().floor()
        dy, dx = py-y0, px-x0
        fixed = self.fixed_samples(x)
        difference = None
        for sy, sx, weight in ((0, 0, (1-dy)*(1-dx)), (1, 1, dy*dx),
                               (0, 1, (1-dy)*dx), (1, 0, dy*(1-dx))):
            indices = torch.cat(((y0+sy).clamp(0, x.shape[-2]-1),
                                 (x0+sx).clamp(0, x.shape[-1]-1)), dim=-1).long()
            neighbor = self._get_x_q(x, indices, n)
            term = (neighbor-fixed) * weight.to(x.dtype).unsqueeze(1)
            difference = term if difference is None else difference+term
        return self.aggregate(self._reshape_x_offset(difference, n))


class GeometricResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=1, rep_csp=False,
                 ldconv_num_param=3, geometry_gain=0.1):
        super().__init__()
        self.shared = (RepCSP(in_c, out_c, num_blocks) if rep_csp else
                       VajraV2InnerBlock(in_c, out_c, num_bottleneck_blocks=num_blocks,
                                        rep_bottleneck=False, shortcut=True))
        self.regular_head = ConvBNAct(out_c, out_c, 1, 3)
        self.geometry = GeometricDifference(out_c, ldconv_num_param)
        self.geometry_scale = ResidualScale(geometry_gain)

    def forward(self, x):
        t = self.shared(x)
        return self.regular_head(t) + self.geometry_scale(self.geometry(t))


class GeometryResidualSAMX(ShapeAdaptiveMerudandaX):
    """Keep split/cascade/concat layout and P4; replace gated fusion by differences.

    This intentionally uses one shared trunk per stage. It is a complete new
    variant, not an isolated test of only the old gate or only trunk sharing.
    """
    def __init__(self, *args, geometry_gain=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < geometry_gain <= 1:
            raise ValueError("geometry_gain must be in (0, 1]")
        self.route = nn.Identity()
        common = dict(num_blocks=self.num_blocks, rep_csp=self.rep_csp,
                      ldconv_num_param=self.ldconv_num_param, geometry_gain=geometry_gain)
        self.block2 = GeometricResidualBlock(self.hidden_c, self.mid_c2, **common)
        self.block3 = GeometricResidualBlock(self.mid_c2, self.mid_c2, **common)
        self.geometry_gain_init = geometry_gain

    def forward(self, x):
        f1, f2 = self.conv1(x).chunk(2, 1)
        y2 = self.block2(f2)
        y3 = self.block3(y2)
        return self.conv4(torch.cat((f1, f2, y2, y3), 1))

    def forward_split(self, x):
        return self.forward(x)

    def get_module_info(self):
        return "GeometryResidualSAMX", f"[{self.in_c}, {self.out_c}, gamma_init={self.geometry_gain_init}]"
