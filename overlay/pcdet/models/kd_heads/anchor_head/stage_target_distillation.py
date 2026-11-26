"""Training-only same-proposal refinement targets for CLEAN."""
import torch

from ....ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from .component_distillation import aligned_iou


@torch.no_grad()
def select_targets(batch, mode, margin, audit=None):
    if mode not in ("late", "quality", "random", "early", "matched_gt", "count_matched"):
        raise ValueError(f"Unknown stage target mode: {mode}")
    early = batch["rois_new_tea"][..., :7]
    late = batch["batch_box_preds_tea"][..., :7]
    targets = late.clone()
    eligible = torch.zeros_like(batch["select_mask"], dtype=torch.bool)
    replaced = torch.zeros_like(eligible)
    reference = torch.zeros_like(eligible) if audit is not None else None
    seed = sum((i + 1) * int(frame) for i, frame in enumerate(batch["frame_id"]))
    generator = torch.Generator(device=early.device).manual_seed(seed)
    for b, gt in enumerate(batch["gt_boxes"]):
        for cls, threshold in enumerate((.7, .5, .5), 1):
            indices = (batch["select_mask"][b].bool() & (batch["teacher_stage_labels"][b] == cls)).nonzero().flatten()
            boxes = gt[gt[:, -1] == cls, :7].contiguous()
            if not indices.numel() or not boxes.numel():
                continue
            a, z = early[b, indices].contiguous(), late[b, indices].contiguous()
            first = boxes_iou3d_gpu(a, boxes).clamp(0, 1)
            second = boxes_iou3d_gpu(z, boxes).clamp(0, 1)
            key = torch.maximum(first, second).argmax(1, keepdim=True)
            x, y = (v.gather(1, key).flatten() for v in (first, second))
            good = x >= threshold
            passed = good | (y >= threshold)
            if reference is not None:
                reference[b, indices] = passed
            if mode == "count_matched":
                # The gate sets the quota, while individual IoUs do not rank the draw.
                quota = indices[passed].numel()
                order = torch.randperm(indices.numel(), device=early.device, generator=generator)
                eligible[b, indices[order[:quota]]] = True
            else:
                eligible[b, indices] = passed
            if mode == "early":
                targets[b, indices[passed]] = a[passed]
                replaced[b, indices] = passed
            elif mode == "matched_gt":
                targets[b, indices[passed]] = boxes[key.flatten()[passed]]
                replaced[b, indices] = passed
            elif mode in ("quality", "random"):
                changes = indices[good & (x - y >= margin)]
                if mode == "random":
                    choices = indices[good]
                    order = torch.randperm(len(choices), device=early.device, generator=generator)
                    changes = choices[order[:len(changes)]]
                targets[b, changes] = early[b, changes]
                replaced[b, changes] = True
    if audit is not None:
        audit.update(stage_gated_pairs=reference.sum().item(),
                     stage_selected=batch["select_mask"].sum().item(),
                     stage_sampled_outside_gate=(eligible & ~reference).sum().item())
    return targets, eligible, replaced


def stage_loss(batch, mode, margin, audit=False):
    details = {} if audit else None
    targets, eligible, replaced = select_targets(batch, mode, margin, details)
    student = batch["xgd_student_boxes"][..., :7][eligible]
    overlaps = aligned_iou(student, targets[eligible]) if len(student) else student.new_empty(0)
    loss = (1 - overlaps).mean() if len(overlaps) else student.sum() * 0
    metrics = dict(stage_loss=loss.detach().item(), stage_pairs=eligible.sum().item(),
                   stage_replaced=replaced.sum().item())
    if audit:
        metrics.update(details, stage_zero_iou=(overlaps == 0).sum().item())
    return loss, metrics
