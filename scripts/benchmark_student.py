"""Batch-1 FP32 latency of the deployed student, including decode and NMS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import load_data_to_gpu
from deployment import load_detector, numerical_protocol


def summarize(samples):
    return {f"p{percentile}_ms": float(np.percentile(samples, percentile)) for percentile in (50, 95)}


def benchmark(args):
    project = Path(__file__).resolve().parents[1]
    os.chdir(project / "CLEAN-main/tools")
    cfg_from_yaml_file(str(args.config), cfg)
    dataset, loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1, False,
                                         workers=0, training=False)
    if len(dataset) < args.warmup + args.frames:
        raise ValueError("insufficient validation frames")
    model, _ = load_detector(dataset, args.weights)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runs, ids = [], []
    with torch.inference_mode():
        for repetition in range(3):
            loader_iter = iter(loader)
            elapsed, prep, current_ids = [], [], []
            for step in range(args.warmup + args.frames):
                start = perf_counter()
                batch = next(loader_iter)
                batch.pop("gt_boxes", None)
                load_data_to_gpu(batch)
                torch.cuda.synchronize()
                ready = perf_counter()
                model(batch)
                torch.cuda.synchronize()
                end = perf_counter()
                if step >= args.warmup:
                    prep.append(1000 * (ready - start))
                    elapsed.append(1000 * (end - ready))
                    current_ids.append(str(batch["frame_id"][0]))
            if repetition and current_ids != ids:
                raise ValueError("frame order changed")
            ids = current_ids
            runs.append(dict(model=summarize(elapsed), preparation=summarize(prep),
                             model_ms=elapsed, preparation_ms=prep))
    pooled = [ms for run in runs for ms in run["model_ms"]]
    end_to_end = [model_ms + prep_ms for run in runs
                  for model_ms, prep_ms in zip(run["model_ms"], run["preparation_ms"])]
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, precision="fp32",
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                parameters=sum(p.numel() for p in model.parameters()), frames=ids, runs=runs,
                warmup=args.warmup, pooled=summarize(pooled), pooled_end_to_end=summarize(end_to_end),
                numerical_protocol=numerical_protocol(), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    args.config, args.weights, args.output = [p.resolve() for p in (args.config, args.weights, args.output)]
    args.output.write_text(json.dumps(benchmark(args), allow_nan=False, indent=2) + "\n")
