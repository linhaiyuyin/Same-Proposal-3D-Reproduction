"""Recompute KITTI AP from a local prediction pickle and its matching infos."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def recompute(infos_path, predictions_path):
    infos_bytes = infos_path.read_bytes()
    prediction_bytes = predictions_path.read_bytes()
    infos = pickle.loads(infos_bytes)
    predictions = pickle.loads(prediction_bytes)
    ids = [v["point_cloud"]["lidar_idx"] for v in infos]
    pred_ids = [v["frame_id"] for v in predictions]
    if len(set(ids)) != len(ids) or ids != pred_ids:
        raise ValueError("prediction frame IDs or their order differ from ground truth")
    report, metrics = get_official_eval_result([copy.deepcopy(i["annos"]) for i in infos],
                                               predictions, CLASSES)
    values = {k: float(v) for k, v in metrics.items()}
    mean = np.mean([values[f"{cls}_3d/moderate_R40"] for cls in CLASSES])
    return report, dict(frames=len(ids), infos_sha256=hashlib.sha256(infos_bytes).hexdigest(),
                        prediction_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
                        moderate_3d_map_r40=float(mean), metrics=values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infos", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report, results = recompute(args.infos, args.predictions)
    print(report)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
