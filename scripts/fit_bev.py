"""Fit a fixed-mask first-BEV-convolution compensation and assess it on held-out training frames."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils.pruning_utils.prune_plain_model import pruning
from bootstrap import ROOT, digest
from channel_layout import parameters
from match_training_objects import object_rows, THRESHOLDS

BASE = ROOT / "CLEAN-main/tools/cfgs/kitti_diagnostics/second-x0.40-aligned.yaml"
WEIGHTS = ROOT / "storage/checkpoints/kitti/second/checkpoint_epoch_80.tensors.pth"
MASK = ROOT / "CLEAN-main/tools/measure_doc/SECONDNet_kitti_mask.txt"

KEY = "backbone_2d.blocks.0.1"
BN = "backbone_2d.blocks.0.2"
RIDGE = 0.1


def features(model, data, idx):
    example = data.collate_batch([data[idx]])
    example.pop("gt_boxes", None)
    load_data_to_gpu(example)
    for module in (model.vfe, model.backbone_3d, model.map_to_bev_module):
        example = module(example)
    return example["spatial_features"], example["spatial_features_stride"]


def cells(data, idx, x, stride, rng):
    boxes = data.kitti_infos[idx]["annos"]["gt_boxes_lidar"]
    floor = np.asarray(data.point_cloud_range[:2])
    step = np.asarray(data.voxel_size[:2]) * stride
    _, _, h, w = x.shape
    foreground = np.zeros((h, w), dtype=bool)
    for box in boxes:
        center = (box[:2] - floor) / step
        # A padded axis-aligned region includes class centers and their context.
        radius = np.linalg.norm(box[3:5]) / (2 * step) + 1
        (x0, y0), (x1, y1) = np.floor(center-radius).astype(int), np.ceil(center+radius).astype(int)+1
        if x1 > 0 and y1 > 0 and x0 < w and y0 < h:
            foreground[max(0,y0):min(h,y1), max(0,x0):min(w,x1)] = True
    occupied = x[0].abs().amax(0).cpu().numpy() > 0
    pools = (np.flatnonzero(foreground), np.flatnonzero(occupied & ~foreground))
    return torch.as_tensor(np.concatenate([rng.choice(p, min(256,len(p)), replace=False) for p in pools]),
                           device=x.device, dtype=torch.long)


def patch_matrix(x, ids):
    _, channels, _, w = x.shape
    row, col = ids.div(w, rounding_mode="floor"), ids.remainder(w)
    padded = F.pad(x[0], (1,1,1,1))
    offsets = torch.arange(3, device=x.device)
    return padded[:, row[:,None,None]+offsets[None,:,None],
                  col[:,None,None]+offsets[None,None,:]].permute(1,0,2,3).reshape(len(ids), channels*9)


def moments():
    return [0, 0, 0]


def update(stat, y):
    y = y[0].flatten(1)
    stat[0] += y.sum(1, dtype=torch.float64).cpu()
    stat[1] += y.square().sum(1, dtype=torch.float64).cpu()
    stat[2] += y.shape[1]


def assign_bn(model, stat):
    bn = model.get_submodule(BN)
    mean = stat[0] / stat[2]
    variance = (stat[1] / stat[2] - mean.square()).clamp_min(0)
    bn.running_mean.copy_(mean)
    bn.running_var.copy_(variance)


def head_outputs(model, data, idx):
    batch = data.collate_batch([data[idx]])
    batch.pop("gt_boxes", None)
    load_data_to_gpu(batch)
    for module in model.module_list:
        batch = module(batch)
    head = model.dense_head.forward_ret_dict
    return {k: head[k] for k in ("cls_preds", "box_preds", "dir_cls_preds")}


def frame_folds(manifest):
    selection = json.loads(manifest.read_text())
    root = ROOT / "storage/datasets/kitti"
    if digest(root / "kitti_infos_pilot_train.pkl") != selection["infos_sha256"]:
        raise ValueError("calibration manifest belongs to different preprocessed inputs")
    pool = set((root / "ImageSets/pilot_train.txt").read_text().split())
    folds = [sorted(frame for subset in selection["strata"].values() for frame in subset[i::3])
             for i in range(3)]
    ids = {frame for fold in folds for frame in fold}
    if len(ids) != 96 or sum(map(len, folds)) != len(ids) or ids != set(selection["frame_ids"]) or ids - pool:
        raise ValueError("calibration needs 96 distinct frames from the training pool")
    return folds


def localization_coverage(data, model, lookup, frames):
    predictions = []
    for start in range(0, len(frames), 4):
        batch = data.collate_batch([data[lookup[frame]] for frame in frames[start:start + 4]])
        batch.pop("gt_boxes", None)
        load_data_to_gpu(batch)
        detected, _ = model(batch)
        predictions.extend(data.generate_prediction_dicts(batch, detected, cfg.CLASS_NAMES))
    rows = list(object_rows([data.kitti_infos[lookup[frame]] for frame in frames], predictions))
    by_class = {}
    for category in THRESHOLDS:
        selected = [row for row in rows if row[2] == category]
        if not selected:
            raise ValueError(f"held-out calibration fold misses {category}")
        by_class[category] = sum(row[-1] for row in selected) / len(selected)
    return dict(by_class=by_class, macro=float(np.mean(list(by_class.values()))))


@torch.no_grad()
def run(manifest, output):
    folds = frame_folds(manifest)
    fit_ids = folds[0]+folds[1]
    held_out = folds[2]
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT/"CLEAN-main/tools")
    cfg_from_yaml_file(str(BASE), cfg)
    cfg.DATA_CONFIG.DATA_SPLIT.test = "pilot_train"
    cfg.DATA_CONFIG.INFO_PATH.test = ["kitti_infos_pilot_train.pkl"]
    data = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False)
    lookup = {r["point_cloud"]["lidar_idx"]:i for i,r in enumerate(data.kitti_infos)}
    torch.manual_seed(665)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dense = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), data)
    dense._load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["model_state"], strict=True)
    layers = list(parameters(dense))
    narrow, _ = pruning(deepcopy(cfg), dense, data)
    raw = torch.as_tensor(np.loadtxt(MASK), device="cuda")
    raw = raw.sum(0) if raw.ndim == 2 else raw
    start = layers[12][3]
    width = layers[12][4]
    choose = raw.argsort(descending=True)[:1382]
    keep_outputs = choose[(choose >= start) & (choose < start+width)]-start
    dense.cuda().eval()
    narrow.cuda().eval()
    layer = narrow.get_submodule(KEY)
    if (layer.kernel_size, layer.stride, layer.padding, layer.groups, layer.bias) != ((3,3),(1,1),(1,1),1,None):
        raise ValueError("calibration assumes one ungrouped 3x3 stride-1 BEV convolution")
    if keep_outputs.numel() != layer.out_channels:
        raise ValueError("dense and narrow output channels are misaligned")
    rng = np.random.default_rng(665)
    w = layer.weight.flatten(1).T
    gram = torch.zeros((len(w),len(w)), dtype=torch.float64)
    cross = torch.zeros(w.shape, dtype=torch.float64)
    sample_count = 0
    for frame in fit_ids:
        idx = lookup[frame]
        x, stride = features(narrow, data, idx)
        y, _ = features(dense, data, idx)
        ids = cells(data, idx, x, stride, rng)
        patches = patch_matrix(x, ids)
        ref = dense.get_submodule(KEY)(y)[0, keep_outputs].flatten(1)[:,ids].T
        baseline = layer(x)[0].flatten(1)[:,ids].T
        torch.testing.assert_close(patches @ w, baseline, atol=1e-4, rtol=1e-4)
        residual = ref-baseline
        gram.add_((patches.T @ patches).cpu())
        cross.add_((patches.T @ residual).cpu())
        sample_count += len(ids)
    penalty = RIDGE*gram.diag().mean().item()
    if sample_count == 0 or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("empty or invalid ridge calibration")
    delta = torch.linalg.solve(gram + penalty*torch.eye(len(w), dtype=gram.dtype),cross)
    if not torch.isfinite(delta).all():
        raise ValueError("non-finite compensation")
    corrected_weight = layer.weight + delta.T.reshape(layer.weight.shape).to(layer.weight)

    stats_original, stats_corrected = moments(), moments()
    for frame in fit_ids:
        x, _ = features(narrow,data,lookup[frame])
        update(stats_original,layer(x))
        update(stats_corrected,F.conv2d(x,corrected_weight,padding=1))
    variants = {}
    raw_weight = layer.weight.detach().clone()
    for name, new_weight, stats in (("aligned",raw_weight,None),
                                    ("bn",raw_weight,stats_original),
                                    ("reconstructed",corrected_weight,stats_corrected)):
        path=output/name
        path.mkdir()
        layer.weight.copy_(new_weight)
        if stats is not None:
            assign_bn(narrow,stats)
            state = narrow.state_dict()
            names = [f"{BN}.running_mean", f"{BN}.running_var"]
            if name=="reconstructed":
                names.append(f"{KEY}.weight")
            torch.save(dict(model_state={k:state[k].cpu() for k in names}),path/"init-patch.pth")
        errors = {key: [] for key in ("cls_preds", "box_preds", "dir_cls_preds")}
        for frame in held_out:
            reference = head_outputs(dense, data, lookup[frame])
            candidate = head_outputs(narrow, data, lookup[frame])
            for key, value in reference.items():
                relative_error = (value - candidate[key]).norm() / value.norm().clamp_min(1e-8)
                errors[key].append(float(relative_error))
        variants[name] = dict(
            coverage=localization_coverage(data, narrow, lookup, held_out),
            head_mean_relative_l2={key: float(np.mean(values)) for key, values in errors.items()},
        )

    report = dict(samples=sample_count, fit_frames=len(fit_ids), held_out_frames=len(held_out),
                  ridge=RIDGE, penalty=penalty, checkpoint_sha256=digest(WEIGHTS), mask_sha256=digest(MASK),
                  manifest_sha256=digest(manifest), variants=variants)
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"initialization already exists at {output}")
    with tempfile.TemporaryDirectory(prefix=".fit-bev-", dir=output.parent) as work:
        staged = Path(work) / output.name
        run(args.manifest.resolve(), staged)
        staged.rename(output)
