"""Score the fixed SECOND calibration frames using one shared output gate per channel."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from channel_layout import parameters


def attach_gates(model):
    layers = list(parameters(model))
    modules = dict(model.named_modules())
    gates, hooks, sizes = [], [], {}
    for name, _, _, start, width in layers:
        stem, position, _ = name.rsplit(".", 2)
        bn = modules[f"{stem}.{int(position) + 1}"]
        if not isinstance(bn, nn.modules.batchnorm._BatchNorm) or bn.num_features != width:
            raise ValueError(f"unexpected normalization after {name}")
        gate = torch.ones(width, device="cuda", requires_grad=True)
        gates.append(gate)

        def apply(module, inputs, value, gate=gate, start=start):
            sizes[start] = value.numel() // gate.numel() * value.element_size()
            shape = (1, -1, *([1] * (value.ndim - 2)))
            return value * gate.view(shape)

        hooks.append(bn.register_forward_hook(apply))
    return layers, gates, hooks, sizes


def compute(manifest, output, limit):
    project = Path(__file__).resolve().parents[1]
    os.chdir(project / "CLEAN-main/tools")
    cfg_from_yaml_file("cfgs/kitti_models/second.yaml", cfg)
    cfg.DATA_CONFIG.DATA_SPLIT.test = "pilot_train"
    cfg.DATA_CONFIG.INFO_PATH.test = ["kitti_infos_pilot_train.pkl"]
    data = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False)
    lookup = {v["point_cloud"]["lidar_idx"]: i for i, v in enumerate(data.kitti_infos)}
    ids = json.loads(manifest.read_text())["frame_ids"]
    if set(ids) - lookup.keys() or len(ids) != len(set(ids)):
        raise ValueError("invalid calibration split")
    if limit:
        ids = ids[:limit]
    torch.manual_seed(665)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), data)
    weights = project / "storage/checkpoints/kitti/second/checkpoint_epoch_80.tensors.pth"
    model._load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model_state"], strict=True)
    model.cuda().train().requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    layers, gates, hooks, sizes = attach_gates(model)
    results, costs, counts = [], [], []
    for frame in ids:
        batch = data.collate_batch([data[lookup[frame]]])
        load_data_to_gpu(batch)
        ret, _, _ = model(batch)
        head = model.dense_head
        labels = head.forward_ret_dict["box_cls_labels"]
        logits = head.forward_ret_dict["cls_preds"].reshape(1, -1, head.num_class)
        losses = [ret["loss"].mean()]
        npos = []
        # Conditional positive-anchor focal losses diagnose each class separately.
        # The main score differentiates the complete detector loss including negatives and regression.
        for cls in range(head.num_class):
            keep = labels == cls + 1
            npos.append(int(keep.sum()))
            if npos[-1]:
                selected = logits[keep].unsqueeze(0)
                target = torch.zeros_like(selected)
                target[..., cls] = 1
                losses.append(head.cls_loss_func(selected, target, weights=torch.ones_like(selected[..., 0])).sum() / npos[-1])
            else:
                losses.append(None)
        values = []
        for index, loss in enumerate(losses):
            if loss is None:
                values.append(np.zeros(3456))
            else:
                gradients = torch.autograd.grad(loss, gates, retain_graph=index < len(losses) - 1)
                values.append(torch.cat(gradients).detach().double().square().cpu().numpy())
        results.append(values)
        costs.append(np.concatenate([np.full(width, sizes[start]) for _, _, _, start, width in layers]))
        counts.append(npos)
        del batch, ret, losses, loss, logits
    for hook in hooks:
        hook.remove()
    results, costs, counts = map(np.asarray, (results, costs, counts))
    if not np.isfinite(results).all() or np.any(costs <= 0) or np.max(results[:, 0]) == 0:
        raise ValueError("invalid Fisher derivatives or costs")
    # Ratio of expected per-frame Fisher information to expected activation bytes per channel.
    scores = 0.5 * results.sum(axis=0) / costs.sum(axis=0)
    output.mkdir(exist_ok=False, parents=True)
    np.savez_compressed(output / "frame-gradients.npz", frame_ids=ids, squared_gradients=results,
                        activation_bytes=costs, positive_counts=counts)
    with (output / "channels.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(("index", "parameter", "channel", "Fisher", *cfg.CLASS_NAMES, "mean_activation_bytes"))
        for name, _, _, start, width in layers:
            for j in range(width):
                i = start + j
                w.writerow((i, name, j, *scores[:, i], costs[:, i].mean()))
    (output / "calibration.json").write_text(json.dumps(dict(frame_ids=ids, classes=cfg.CLASS_NAMES,
        positive_counts=counts.sum(0).tolist(), width=scores.shape[-1],
        checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        calibration_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    compute(args.manifest.resolve(), args.output.resolve(), args.limit)
