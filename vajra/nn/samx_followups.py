"""Independent SAMX follow-ups; earlier implementations remain unchanged."""
from copy import deepcopy
import math

import torch
from torch import nn

from vajra.nn.modules import ConvBNAct
from vajra.nn.samx_geometry import (GeometricDifference, GeometryResidualSAMX,
                                    ResidualScale)


class DualPathGeometryBlock(nn.Module):
    """Split a shared stage into identically initialized independent trunks."""
    def __init__(self, shared_block):
        super().__init__()
        self.shared = shared_block.shared  # retain regular trunk state-dict names
        self.deform_shared = deepcopy(shared_block.shared)
        self.regular_head = shared_block.regular_head
        self.geometry = shared_block.geometry
        self.geometry_scale = shared_block.geometry_scale

    def forward(self, x):
        regular = self.regular_head(self.shared(x))
        correction = self.geometry(self.deform_shared(x))
        return regular+self.geometry_scale(correction)


class DualPathGeometrySAMX(GeometryResidualSAMX):
    """Only split the trunks; preserve all common initialization and P4 layout."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.block2 = DualPathGeometryBlock(self.block2)
        self.block3 = DualPathGeometryBlock(self.block3)

    def get_module_info(self):
        return 'DualPathGeometrySAMX', f'[{self.in_c}, {self.out_c}, gamma_init={self.geometry_gain_init}]'


class StructuredGeometricDifference(GeometricDifference):
    """Rotate/stretch the same three-point template around its fixed centroid.

    Three predicted fields: angle, y scale, x scale. No independent point drift.
    Coordinates and transforms remain FP32 under AMP. Gain initialization is
    handled by the surrounding residual module, independently of this transform.
    """
    def __init__(self, channels, num_param=3):
        super().__init__(channels,num_param)
        if num_param!=3:
            raise ValueError('This controlled trial uses the original three-point template')
        self.angle_limit = math.pi
        self.scale_limit = 2.0
        self.p_conv = nn.Conv2d(channels,3,3,padding=1)
        nn.init.zeros_(self.p_conv.weight)
        nn.init.zeros_(self.p_conv.bias)

    def _get_p(self, parameters, dtype):
        parameters = parameters.float()
        n,h,w = self.num_param,parameters.shape[-2],parameters.shape[-1]
        template = self.p_n.to(device=parameters.device,dtype=torch.float32)
        qy,qx = template[:,:n],template[:,n:]
        cy,cx = qy-qy.mean(1,keepdim=True),qx-qx.mean(1,keepdim=True)
        angle = self.angle_limit*parameters[:,:1].tanh()
        sy = (math.log(self.scale_limit)*parameters[:,1:2].tanh()).exp()
        sx = (math.log(self.scale_limit)*parameters[:,2:3].tanh()).exp()
        cosine,sine = angle.cos(),angle.sin()
        # Express the transform as offsets. All coefficients are exactly zero
        # at initialization, avoiding centroid round-off in the identity case.
        dy = (cosine*sy-1)*cy-sine*sx*cx
        dx = sine*sy*cy+(cosine*sx-1)*cx
        return self._get_p_0(h,w,n,dtype)+template+torch.cat((dy,dx),1)

    def get_module_info(self):
        return 'StructuredGeometricDifference', f'[{self.inc}, N=3, angle=pi, scale=(0.5,2)]'


class ProtoGeometryEnhance(nn.Module):
    """P3-only residual before Proto; detection/mask coefficient inputs are intact."""
    def __init__(self, channels, hidden=16, mode='free', gain=0.1):
        super().__init__()
        if mode not in ('free','structured'):
            raise ValueError(mode)
        if not 0<gain<=1 or isinstance(gain,bool):
            raise ValueError('gain must be in (0,1]')
        if isinstance(hidden,bool) or not 0<hidden<=channels:
            raise ValueError('hidden must be a positive integer <= input channels')
        self.mode,self.hidden,self.in_channels = mode,hidden,channels
        self.down = ConvBNAct(channels,hidden,1,1)
        self.geometry = GeometricDifference(hidden,3)
        self.up = nn.Conv2d(hidden,channels,1,bias=False)
        self.geometry_scale = ResidualScale(gain)
        if mode=='structured':
            # Keep down/up/aggregation initialization identical to free sampling.
            with torch.random.fork_rng(devices=[]):
                structured = StructuredGeometricDifference(hidden,3)
            structured.aggregate.load_state_dict(self.geometry.aggregate.state_dict())
            self.geometry = structured

    def forward(self,x):
        return x+self.geometry_scale(self.up(self.geometry(self.down(x))))


def configure_proto_geometry(head, mode, hidden=16, gain=.1):
    if mode not in ('off','free','structured'):
        raise ValueError(mode)
    if mode=='off':
        return
    if not isinstance(head.proto_enhance,nn.Identity):
        raise ValueError('Proto geometry requires CFSA and CFSAv1 disabled')
    with torch.random.fork_rng(devices=[]):
        module = ProtoGeometryEnhance(head.in_channels[0],hidden,mode,gain)
    from vajra.utils.torch_utils import initialize_weights
    initialize_weights(module)
    p = next(head.proto.parameters())
    head.proto_enhance = module.to(device=p.device,dtype=p.dtype)
