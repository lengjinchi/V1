import io
import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from samx_followup_config import MODES,options
from samx_followup_validation import BoundaryAccumulator
from vajra.configs import get_config
from vajra.core.model import Model
from vajra.core.trainer import Trainer
from vajra.models.vajra.segment.train import SegmentationTrainer
from vajra.nn.modules import LDConv
from vajra.nn.samx_geometry import GeometryResidualSAMX,ResidualScale
from vajra.nn.samx_followups import (DualPathGeometrySAMX,ProtoGeometryEnhance,
                                    StructuredGeometricDifference)


class FixedParams(nn.Module):
    def __init__(self,value):
        super().__init__();self.weight=nn.Parameter(value)
    def forward(self,x):return self.weight


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_dual_common_initialization_independence_and_live_geometry(self):
        torch.manual_seed(42);shared=GeometryResidualSAMX(16,16,16,8,rep_csp=True).eval()
        torch.manual_seed(42);dual=DualPathGeometrySAMX(16,16,16,8,rep_csp=True).eval()
        state=dual.state_dict()
        for k,v in shared.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
        for block in (dual.block2,dual.block3):
            for p,q in zip(block.shared.parameters(),block.deform_shared.parameters()):
                self.assertNotEqual(p.data_ptr(),q.data_ptr());torch.testing.assert_close(p,q,atol=0,rtol=0)
        x=torch.randn(2,16,7,11)
        torch.testing.assert_close(shared(x),dual(x),atol=0,rtol=0)
        dual(x).square().mean().backward()
        self.assertGreater(dual.block2.geometry.p_conv.weight.grad.abs().sum().item(),0)

    def test_proto_identity_live_gradients_and_common_structured_initialization(self):
        torch.manual_seed(42);free=ProtoGeometryEnhance(64,16,'free').eval()
        torch.manual_seed(42);structured=ProtoGeometryEnhance(64,16,'structured').eval()
        for k,v in free.state_dict().items():
            if 'p_conv' not in k:torch.testing.assert_close(v,structured.state_dict()[k],atol=0,rtol=0)
        x=torch.randn(2,64,8,12)
        for m in (free,structured):
            torch.testing.assert_close(m(x),x,atol=0,rtol=0)
            m(x).square().mean().backward()
            grad=m.geometry.p_conv.weight.grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)

    def test_structured_reference_output_and_parameter_input_gradients(self):
        torch.manual_seed(9)
        b,c,h,w,n=2,2,5,7,3
        x=torch.randn(b,c,h,w,requires_grad=True)
        fields=torch.randn(b,3,h,w)*.2
        m=StructuredGeometricDifference(c)
        m.p_conv=FixedParams(fields.clone())
        actual=m(x)
        xr=x.detach().clone().requires_grad_();pr=fields.clone().requires_grad_()
        kr=m.aggregate.weight.detach().clone().requires_grad_()
        q=m.p_n.reshape(1,2,n,1,1).float()
        centroid=q.mean(2,keepdim=True);centered=q-centroid
        theta=math.pi*pr[:,:1].tanh();sy=(math.log(2)*pr[:,1:2].tanh()).exp();sx=(math.log(2)*pr[:,2:3].tanh()).exp()
        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        py=yy[None,None]+centroid[:,0]+theta.cos()*sy*centered[:,0]-theta.sin()*sx*centered[:,1]
        px=xx[None,None]+centroid[:,1]+theta.sin()*sy*centered[:,0]+theta.cos()*sx*centered[:,1]
        def sample(y,z):
            grid=torch.stack((2*z/(w-1)-1,2*y/(h-1)-1),-1).permute(0,2,3,1,4).reshape(b,h,w*n,2)
            return F.grid_sample(xr,grid,padding_mode='border',align_corners=True).reshape(b,c,h,w,n)
        fixed_y=(yy[None,None]+q[:,0]).expand(b,-1,-1,-1)
        fixed_x=(xx[None,None]+q[:,1]).expand(b,-1,-1,-1)
        diff=sample(py,px)-sample(fixed_y,fixed_x)
        packed=diff.permute(0,1,2,4,3).reshape(b,c,h*n,w)
        expected=F.conv2d(packed,kr,stride=(n,1))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        upstream=torch.randn_like(actual)
        (actual*upstream).sum().backward();(expected*upstream).sum().backward()
        for a,e in ((x.grad,xr.grad),(m.p_conv.weight.grad,pr.grad),(m.aggregate.weight.grad,kr.grad)):
            torch.testing.assert_close(a,e,atol=2e-5,rtol=3e-5)

    def test_structured_centroid_identity_constants_and_singletons(self):
        for h,w in ((1,1),(1,7),(5,1),(5,7)):
            m=StructuredGeometricDifference(2)
            x=torch.randn(2,2,h,w)
            self.assertEqual(m(x).count_nonzero().item(),0)
            fields=torch.randn(2,3,h,w)
            p=m._get_p(fields,fields.type())
            base=m._get_p_0(h,w,3,fields.type())+m.p_n
            torch.testing.assert_close((p-base).reshape(2,2,3,h,w).mean(2),torch.zeros(2,2,h,w),atol=4e-7,rtol=0)
            with torch.no_grad():m.p_conv.bias.copy_(torch.tensor([.7,-.5,.3]))
            self.assertEqual(m(torch.full_like(x,3.7)).count_nonzero().item(),0)

    def build_model(self,mode):
        args=get_config(model_configuration=options(mode))
        trainer=SegmentationTrainer.__new__(SegmentationTrainer);trainer.args,trainer.data=args,{'nc':4}
        torch.manual_seed(42)
        model=trainer.get_model('vajra-v1-nano-seg',verbose=False);model.args=args
        return model,args

    def test_model_loss_optimizer_and_pickle_each_variant(self):
        sample={'img':torch.rand(2,3,64,64),'masks':torch.zeros(2,16,16),
            'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
            'bboxes':torch.tensor([[.5,.5,.5,.5]]*2)}
        sample['masks'][:,4:12,4:12]=1
        for mode in MODES:
            model,args=self.build_model(mode)
            loss,_=model(sample);self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertFalse(model.criterion.use_morph_mask_loss)
            trainer=Trainer.__new__(Trainer);trainer.args=args
            optimizer=trainer.build_optimizer(model,name='SGD',lr=.01,decay=.0005)
            ids=[id(p) for g in optimizer.param_groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters()})
            slow={id(p) for m in model.modules() if isinstance(m,LDConv) for p in m.p_conv.parameters()}
            self.assertEqual(slow,{id(p) for g in optimizer.param_groups if g.get('lr_scale')==.1 for p in g['params']})
            for m in model.modules():
                if isinstance(m,LDConv):
                    self.assertTrue(torch.isfinite(m.p_conv.weight.grad).all())
                    self.assertGreater(m.p_conv.weight.grad.abs().sum().item(),0)
                if isinstance(m,ResidualScale):self.assertEqual(ids.count(id(m.weight)),1)
            model.eval()
            with torch.inference_mode():expected=model(sample['img'])
            stream=io.BytesIO();torch.save(model,stream);stream.seek(0)
            reloaded=torch.load(stream,weights_only=False).eval()
            with torch.inference_mode():torch.testing.assert_close(reloaded(sample['img']),expected,atol=0,rtol=0)
            self.assertEqual(Model._reset_checkpoint_args(vars(args))['samx_proto_mode'],args.samx_proto_mode)

    def test_proto_changes_only_prototypes_in_fixed_weight_forward(self):
        model,_=self.build_model('proto_free');model.eval()
        base_args=get_config(model_configuration={**options('proto_free'),'samx_proto_mode':'off'})
        trainer=SegmentationTrainer.__new__(SegmentationTrainer)
        trainer.args,trainer.data=base_args,{'nc':4}
        torch.manual_seed(42)
        baseline=trainer.get_model('vajra-v1-nano-seg',verbose=False).eval()
        enhanced_state=model.state_dict()
        for k,v in baseline.state_dict().items():
            torch.testing.assert_close(v,enhanced_state[k],atol=0,rtol=0)
        x=torch.rand(2,3,96,96)
        with torch.inference_mode():
            before=model(x)
            torch.testing.assert_close(before,baseline(x),atol=0,rtol=0)
            model.model[-1].proto_enhance.geometry.p_conv.bias.fill_(.3)
            after=model(x)
        torch.testing.assert_close(before[0],after[0],atol=0,rtol=0)
        self.assertGreater((before[1][-1]-after[1][-1]).abs().max().item(),0)

    def test_invalid_and_conflicting_config(self):
        for change in ({'samx_proto_mode':'typo'},{'samx_proto_mode':'free','use_cfsa':True},
                       {'samx_proto_mode':'structured','use_cfsav1':True},
                       {'samx_proto_gain':0},{'samx_proto_channels':True},{'samx_proto_channels':0}):
            with self.assertRaises((TypeError,ValueError)):get_config(model_configuration=change)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_amp_and_half_each_variant(self):
        for mode in MODES:
            model,_=self.build_model(mode);model=model.cuda().train()
            sample={'img':torch.rand(2,3,64,64,device='cuda'),'masks':torch.zeros(2,16,16,device='cuda'),
                'batch_idx':torch.tensor([0.,1.],device='cuda'),'cls':torch.zeros(2,1,device='cuda'),
                'bboxes':torch.tensor([[.5,.5,.5,.5]]*2,device='cuda')}
            sample['masks'][:,4:12,4:12]=1
            with torch.autocast('cuda',dtype=torch.float16):loss,_=model(sample)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            for m in model.modules():
                if isinstance(m,LDConv):self.assertTrue(torch.isfinite(m.p_conv.weight.grad).all())
            model.half().eval()
            with torch.inference_mode():self.assertTrue(torch.isfinite(model(sample['img'].half())[0]).all())


class BoundaryTests(unittest.TestCase):
    def masks(self):
        x=torch.zeros(1,12,12,dtype=torch.bool);x[:,3:9,3:9]=1
        return x

    def test_perfect_duplicates_false_positives_and_class_mismatch(self):
        x=self.masks();c=torch.tensor([0]);score=torch.tensor([.9])
        a=BoundaryAccumulator();a.update(x,c,score,x,c)
        self.assertEqual(a.report()['overall']['f1'],1.)
        a=BoundaryAccumulator();a.update(x.expand(2,-1,-1),torch.tensor([0,0]),torch.tensor([.9,.8]),x,c)
        r=a.report()['overall'];self.assertEqual(r['precision'],.5);self.assertEqual(r['recall'],1.)
        self.assertAlmostEqual(r['f1'],2/3)
        a=BoundaryAccumulator();a.update(x,torch.tensor([1]),score,x,c)
        self.assertEqual(a.report()['overall']['f1'],0.)

    def test_unmatched_gt_confidence_and_tolerance(self):
        x=self.masks();gt=torch.cat((x,x),0);c=torch.tensor([0]);score=torch.tensor([.9])
        a=BoundaryAccumulator();a.update(x,c,score,gt,torch.tensor([0,0]))
        r=a.report()['overall'];self.assertEqual(r['precision'],1.);self.assertEqual(r['recall'],.5)
        a=BoundaryAccumulator();a.update(x,c,torch.tensor([.1]),x,c)
        self.assertEqual(a.report()['overall']['matched_instances'],0)
        self.assertEqual(a.report()['overall']['gt_instances'],1)
        shifted=torch.roll(x,1,dims=2)
        a=BoundaryAccumulator();a.update(shifted,c,score,x,c)
        self.assertEqual(a.report()['overall']['f1'],1.)


if __name__=='__main__':unittest.main()
