"""CPU-only synthetic regression test for stage selection and oriented box IoU.
Requires torch and shapely; does not exercise CUDA or OpenPCDet integration.
"""
from pathlib import Path
import importlib.util,sys,types,torch,math,random
from shapely.geometry import Polygon
root=Path(__file__).resolve().parents[1] / 'overlay/pcdet/models/kd_heads/anchor_head'
# Load the exact shipped source without importing CUDA/OpenPCDet.
for name in ['pcdet','pcdet.models','pcdet.models.kd_heads','pcdet.models.kd_heads.anchor_head','pcdet.ops','pcdet.ops.iou3d_nms']:
 m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
base='pcdet.models.kd_heads.anchor_head.'
def load(name):
 spec=importlib.util.spec_from_file_location(base+name,root/(name+'.py')); m=importlib.util.module_from_spec(spec);sys.modules[base+name]=m;spec.loader.exec_module(m);return m
c=load('component_distillation')
def pairwise(a,b):
 n,m=len(a),len(b);return c.aligned_iou(a[:,None,:].expand(n,m,7).reshape(-1,7),b[None,:,:].expand(n,m,7).reshape(-1,7)).reshape(n,m)
stub=types.ModuleType('pcdet.ops.iou3d_nms.iou3d_nms_utils');stub.boxes_iou3d_gpu=pairwise;sys.modules[stub.__name__]=stub
s=load('stage_target_distillation')
torch.manual_seed(2026)
a=torch.randn(256,7,dtype=torch.float64); a[:,3:6]=a[:,3:6].abs()+0.3; b=a+0.35*torch.randn_like(a);b[:,3:6]=b[:,3:6].abs()+0.15
pred=c.aligned_iou(a,b);err=[]
for i in range(len(a)):
 pa,pb=Polygon(c.corners(a[i:i+1])[0].numpy()),Polygon(c.corners(b[i:i+1])[0].numpy());x,y=a[i],b[i];h=max(0,min(x[2]+x[5]/2,y[2]+y[5]/2)-max(x[2]-x[5]/2,y[2]-y[5]/2));v=pa.intersection(pb).area*h;ref=v/(x[3:6].prod()+y[3:6].prod()-v);err.append(abs(pred[i]-ref).item())
assert max(err)<1e-9
print('oriented IoU vs independent Shapely: 256 pairs; max abs error',max(err))
a.requires_grad_(); (1-c.aligned_iou(a,b)).mean().backward();assert torch.isfinite(a.grad).all();assert a.grad.abs().sum()>0
print('finite nonzero student gradients: PASS')
# fixture: early-only pass, late-only pass, both pass, neither, no-GT class, padding, second frame
box=lambda x: [x,0,0,2,2,2,0]
e=torch.tensor([[box(0),box(1),box(0.1),box(8),box(0),box(0)],[box(0),box(1),box(0.1),box(8),box(0),box(0)]],dtype=torch.float32)
l=torch.tensor([[box(1),box(0),box(0.2),box(9),box(0),box(0)],[box(1),box(0),box(0.2),box(9),box(0),box(0)]],dtype=torch.float32)
student=(l+torch.tensor([.05,.01,.02,0.,0.,0.,.02])).requires_grad_();l.requires_grad_();e.requires_grad_()
batch={'rois_new_tea':e,'batch_box_preds_tea':l,'select_mask':torch.tensor([[1,1,1,1,1,0],[1,1,1,1,1,0]]),'teacher_stage_labels':torch.tensor([[1,1,1,1,3,1],[1,1,1,1,3,1]]),'frame_id':['000001','000002'],'gt_boxes':torch.tensor([[box(0)+[1]],[box(0)+[1]]],dtype=torch.float32),'xgd_student_boxes':student}
mask=torch.tensor([[True,True,True,False,False,False]]*2)
for mode in ['early','late','matched_gt']:
 t,m,r=s.select_targets(batch,mode,.05);assert torch.equal(m,mask);assert not t.requires_grad
print('OR gate, same-class GT, missing GT, padding, gated variants: PASS')
for mode in ['late','early','matched_gt','count_matched']:
 student.grad=None;loss,stats=s.stage_loss(batch,mode,.05,audit=True);loss.backward();assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0;assert e.grad is None and l.grad is None;assert stats['stage_pairs']==6
 if mode=='count_matched':
  t,m,r=s.select_targets(batch,mode,.05);assert torch.equal(m.sum(1),mask.sum(1));assert not m[:,4:].any()
print('matched per-frame quota, deterministic draws, detached teacher targets, gradients: PASS')
batch['select_mask'].zero_();student.grad=None;loss,stats=s.stage_loss(batch,'late',.05);loss.backward();assert loss==0 and stats['stage_pairs']==0 and student.grad.abs().sum()==0
print('empty selection: PASS')
