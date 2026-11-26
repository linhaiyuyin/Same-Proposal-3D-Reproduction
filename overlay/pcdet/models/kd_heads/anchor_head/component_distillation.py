"""GT-directed geometric component selection for the LiDAR adaptation of StereoDistill XGD."""
import torch


def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def corners(boxes):
    signs = boxes.new_tensor([[-1.,-1.], [1.,-1.], [1.,1.], [-1.,1.]])
    offsets = signs * boxes[:, None, 3:5] / 2
    c, s = boxes[:, None, 6].cos(), boxes[:, None, 6].sin()
    x, y = offsets.unbind(-1)
    return boxes[:,None,:2] + torch.stack((c*x-s*y,s*x+c*y), -1)


def aligned_iou(a, b):
    """Piecewise-differentiable exact oriented IoU for paired upright positive-size boxes."""
    pa, pb = corners(a), corners(b)
    ea, eb = pa.roll(-1,1)-pa, pb.roll(-1,1)-pb
    inside_a = (cross2d(eb[:,None], pa[:,:,None] - pb[:,None]) >= -1e-7).all(-1)
    inside_b = (cross2d(ea[:,None], pb[:,:,None] - pa[:,None]) >= -1e-7).all(-1)
    offset = pb[:,None] - pa[:,:,None]
    denom = cross2d(ea[:,:,None], eb[:,None])
    nonparallel = denom.abs() > 1e-9
    safe = torch.where(nonparallel, denom, 1.)
    t = cross2d(offset, eb[:,None]) / safe
    u = cross2d(offset, ea[:,:,None]) / safe
    intersections = pa[:,:,None] + t[...,None] * ea[:,:,None]
    on_segments = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    points = torch.cat((pa, pb, intersections.flatten(1,2)), 1)
    keep = torch.cat((inside_a, inside_b, on_segments.flatten(1,2)),1)
    n = keep.sum(1).clamp_min(1)
    center = (points * keep[...,None]).sum(1) / n[:,None]
    relative = points - center[:,None]
    angle = torch.atan2(relative[...,1],relative[...,0]).masked_fill(~keep, 8.)
    order = angle.argsort(1)
    relative = relative.gather(1,order[...,None].expand_as(relative))
    indices = torch.arange(relative.shape[1],device=a.device)[None,:]
    successor = (indices + 1) % n[:,None]
    edges = cross2d(relative,relative.gather(1,successor[...,None].expand_as(relative)))
    area = (edges * (indices < n[:,None])).sum(1).abs() / 2
    bottom = torch.maximum(a[:,2]-a[:,5]/2,b[:,2]-b[:,5]/2)
    top = torch.minimum(a[:,2]+a[:,5]/2,b[:,2]+b[:,5]/2)
    intersection = area * (top-bottom).clamp_min(0)
    return (intersection / (a[:,3:6].prod(1)+b[:,3:6].prod(1)-intersection).clamp_min(1e-8)).clamp(0,1)


def directed_component_loss(student, teacher, gt):
    teacher, gt = teacher.detach(), gt.detach()
    if len(student) == 0:
        return student.sum()*0, student.new_zeros(3)
    ds, dg = teacher - student.detach(), gt - student.detach()
    def angle_difference(a):
        return torch.atan2(a.sin(), a.cos())
    good = torch.stack(((ds[:,:3]*dg[:,:3]).sum(1)>0, (ds[:,3:6]*dg[:,3:6]).sum(1)>0,
                        angle_difference(ds[:,6])*angle_difference(dg[:,6])>0),1)
    use = torch.cat((good[:,:1].expand(-1,3),good[:,1:2].expand(-1,3),good[:,2:3]),1)
    soft = torch.where(use, teacher, student.detach())
    active = torch.where(use, student, student.detach())
    return (1 - aligned_iou(active,soft)).mean(), good.float().mean(0)
