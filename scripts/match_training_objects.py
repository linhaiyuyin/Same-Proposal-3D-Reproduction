"""Produce per-object 3D localization diagnostics for a training-side split."""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}


def pair(iou, threshold):
    """Maximize count above threshold first, then total IoU among same-class boxes."""
    if not iou.size:
        return {}
    bonus = min(iou.shape) + 1
    gains = iou + bonus * (iou >= threshold)
    rows, cols = linear_sum_assignment(gains, maximize=True)
    return {int(r): int(c) for r, c in zip(rows, cols) if iou[r, c] > 0}


def polygon_iou_3d(a, b):
    """Float64 polygon clipping for overlaps outside the physical domain of the CUDA kernel."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    def corners(box):
        c, s = np.cos(box[6]), np.sin(box[6])
        signs = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        return (signs * box[3:5] / 2) @ np.array([[c,s],[-s,c]]) + box[:2]

    polygon, clipping = list(corners(a)), corners(b)
    for start, end in zip(clipping, np.roll(clipping, -1, axis=0)):
        if not polygon:
            return 0.0
        edge = end - start
        def side(point):
            v = point - start
            return edge[0] * v[1] - edge[1] * v[0]
        result = []
        previous = polygon[-1]
        old = side(previous)
        for point in polygon:
            new = side(point)
            if (old >= 0) != (new >= 0):
                result.append(previous + old / (old - new) * (point - previous))
            if new >= 0:
                result.append(point)
            previous, old = point, new
        polygon = result
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon) - polygon[0]
    area = abs(np.dot(points[:,0], np.roll(points[:,1], -1)) -
               np.dot(points[:,1], np.roll(points[:,0], -1))) / 2
    height = max(0., min(a[2]+a[5]/2, b[2]+b[5]/2) - max(a[2]-a[5]/2, b[2]-b[5]/2))
    intersection = area * height
    value = intersection / (np.prod(a[3:6]) + np.prod(b[3:6]) - intersection)
    if not 0 <= value <= 1 + 1e-12:
        raise ValueError("invalid independent polygon intersection")
    return min(float(value), 1.)


def object_rows(infos, predictions):
    ids = [v["point_cloud"]["lidar_idx"] for v in infos]
    if [v["frame_id"] for v in predictions] != ids or len(set(ids)) != len(ids):
        raise ValueError("frame identity/order mismatch")
    for frame, info, predicted in zip(ids, infos, predictions):
        ann = info["annos"]
        for name, threshold in THRESHOLDS.items():
            gt_indices = np.flatnonzero(ann["name"] == name)
            if not len(gt_indices):
                continue
            boxes = ann["gt_boxes_lidar"][ann["index"][gt_indices], :7]
            pred_indices = np.flatnonzero(predicted["name"] == name)
            pred_boxes = predicted["boxes_lidar"][pred_indices, :7]
            iou = np.empty((len(boxes), 0), dtype=np.float32)
            if len(pred_indices):
                tensors = [torch.as_tensor(v, dtype=torch.float32, device="cuda").contiguous()
                           for v in (boxes, pred_boxes)]
                iou = boxes_iou3d_gpu(*tensors).cpu().numpy()
                for r, c in np.argwhere(~np.isfinite(iou) | (iou < 0) | (iou > 1)):
                    original = iou[r,c]
                    iou[r,c] = polygon_iou_3d(boxes[r], pred_boxes[c])
                    print(f"{frame} {name} GT {gt_indices[r]} prediction {pred_indices[c]}: {original} -> {iou[r,c]}")
            pairs = pair(iou, threshold)
            for i, gt in enumerate(gt_indices):
                j = pairs.get(i)
                overlap = float(iou[i, j]) if j is not None else 0.0
                score = float(predicted["score"][pred_indices[j]]) if j is not None else None
                yield (frame, int(gt), name, int(ann["difficulty"][gt]),
                       float(np.hypot(*boxes[i, :2])), int(ann["num_points_in_gt"][gt]),
                       overlap, score, int(overlap >= threshold))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("infos", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with a.infos.open("rb") as f, a.predictions.open("rb") as g, a.output.open("w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(("frame", "gt_index", "class", "difficulty", "range_m", "fov_points",
                         "matched_3d_iou", "prediction_score", "meets_iou_threshold"))
        writer.writerows(object_rows(pickle.load(f), pickle.load(g)))
