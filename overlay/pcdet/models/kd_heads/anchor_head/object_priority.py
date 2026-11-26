"""Training-only supervision weights for the proposals used by CLEAN."""
import torch
from ....ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


@torch.no_grad()
def object_weights(batch, mode):
    proposals = batch['rois_new_tea']
    valid = batch['select_mask'].bool()
    labels = batch['cls_preds_selected_kd'].reshape(*valid.shape, -1).argmax(-1) + 1
    priority = torch.zeros_like(valid, dtype=proposals.dtype)
    reliability = torch.zeros_like(priority)
    matched = torch.zeros_like(valid)
    donors = torch.zeros_like(valid)
    targets = torch.zeros_like(proposals)
    thresholds = proposals.new_tensor([0., 0.7, 0.5, 0.5])
    for i, teacher in enumerate(batch['decoded_pred_tea']):
        gt = batch['gt_boxes'][i]
        present = gt[:, -1] > 0
        if not present.any() or not valid[i].any():
            continue
        boxes = gt[present, :7].contiguous()
        classes = gt[present, -1].long()
        context = batch['gt_kd_context'][i, present]
        indices = valid[i].nonzero().flatten()
        iou = boxes_iou3d_gpu(proposals[i, indices, :7].contiguous(), boxes).clamp(0, 1)
        iou.masked_fill_(labels[i, indices, None] != classes[None, :], 0)
        best, gt_ids = iou.max(-1)
        selected = best >= 0.10
        indices, gt_ids = indices[selected], gt_ids[selected]
        if not indices.numel():
            continue
        q = torch.zeros_like(classes, dtype=proposals.dtype)
        if len(teacher['pred_boxes']):
            iou = boxes_iou3d_gpu(boxes, teacher['pred_boxes'][:, :7].contiguous()).clamp(0, 1)
            iou.masked_fill_(classes[:, None] != teacher['pred_labels'][None, :], 0)
            q = iou.max(-1).values
            q = q * (q >= thresholds[classes])
        matched[i, indices] = True
        targets[i, indices] = boxes[gt_ids]
        priority[i, indices] = context[gt_ids, 0]
        reliability[i, indices] = q[gt_ids]
        donors[i, indices] = context[gt_ids, 1] != int(batch['frame_id'][i])
    if mode == 'teacher':
        extra = reliability
    elif mode == 'degradation':
        extra = priority
    elif mode in ('joint', 'shuffle'):
        extra = reliability * priority
    elif mode == 'class_joint':
        extra = reliability * torch.where(labels == 1, priority, 1.)
    elif mode == 'xgd':
        extra = torch.zeros_like(priority)
    else:
        raise ValueError(f'unknown proposal weight mode {mode}')
    if mode == 'shuffle':
        values = extra[matched]
        gen = torch.Generator(device=values.device)
        gen.manual_seed(665 + sum((i+1)*int(frame) for i, frame in enumerate(batch['frame_id'])))
        extra[matched] = values[torch.randperm(values.numel(), device=values.device, generator=gen)]
    weights = (1 + 2 * extra) * valid
    weights *= valid.sum() / weights.sum().clamp_min(1.)
    stats = dict(kd_valid=valid.sum(), kd_gt_matched=matched.sum(), kd_donors=donors.sum(),
                 kd_reliable=(reliability > 0).sum(), kd_prior_sum=(priority * matched).sum(),
                 kd_extra_sum=extra.sum(), kd_extra_donor=(extra * donors).sum(),
                 kd_weight_sum=weights.sum())
    return weights.reshape(-1, 1), stats, targets, matched
