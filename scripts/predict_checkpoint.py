"""Load a deployment checkpoint, run KITTI without GT input and archive predictions."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import load_data_to_gpu

from deployment import load_detector, numerical_protocol
from reevaluate import recompute


def predict(args):
    project = Path(__file__).resolve().parents[1]
    os.chdir(project / "CLEAN-main/tools")
    cfg_from_yaml_file(str(args.config), cfg)
    cfg.DATA_CONFIG.DATA_SPLIT.test = args.split
    cfg.DATA_CONFIG.INFO_PATH.test = [f"kitti_infos_{args.split}.pkl"]
    dataset, loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch, False,
                                         workers=args.workers, training=False)
    model, progress = load_detector(dataset, args.weights)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("gt_boxes", None)
            load_data_to_gpu(batch)
            detected, _ = model(batch)
            results.extend(dataset.generate_prediction_dicts(batch, detected, cfg.CLASS_NAMES))
    predictions = args.output / "result.pkl"
    predictions.write_bytes(pickle.dumps(results))
    infos = dataset.root_path / cfg.DATA_CONFIG.INFO_PATH.test[0]
    report, metrics = recompute(infos, predictions)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (args.output / "evaluation.txt").write_text(report + "\n")
    evidence = dict(config=str(args.config), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    weights=str(args.weights), numerical_protocol=numerical_protocol(), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                    deployed_parameters=sum(p.numel() for p in model.parameters()), split=args.split,
                    frames=len(results), batch=args.batch, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                    **progress)
    (args.output / "inference.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"{args.split}: {len(results)} frames, mAP {metrics['moderate_3d_map_r40']:.6f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--split", choices=("val", "pilot_val", "pilot_train", "train"), default="val")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.batch < 1 or a.workers < 0:
        p.error("invalid loader parameters")
    a.config, a.weights, a.output = (path.resolve() for path in (a.config, a.weights, a.output))
    predict(a)
