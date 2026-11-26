"""Select a shared, class-stratified calibration set from pilot_train before scoring."""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


def select(infos, per_class):
    groups = {name: [] for name in ("Cyclist", "Pedestrian", "Car")}
    annotations = {}
    for frame in infos:
        key = frame["point_cloud"]["lidar_idx"]
        counts = Counter(frame["annos"]["name"])
        if key in annotations:
            raise ValueError(f"duplicate frame {key}")
        annotations[key] = counts
        for name, pool in groups.items():
            if counts[name]:
                pool.append(key)
                break
    strata = {}
    for name, ids in groups.items():
        if len(ids) < per_class:
            raise ValueError(f"too few distinct frames containing {name}")
        ordered = sorted(ids, key=lambda key: hashlib.sha256(("chapter3-cal-v1/" + key).encode()).digest())
        strata[name] = ordered[:per_class]
    selection = sorted(key for subset in strata.values() for key in subset)
    counts = {name: sum(annotations[key][name] for key in selection) for name in groups}
    return dict(frame_ids=selection, strata=strata, gt_counts=counts, pool_frames=len(annotations))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilot_train_infos", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--per-class", type=int, default=32)
    args = p.parse_args()
    if args.per_class <= 0:
        p.error("per-class must be positive")
    data = args.pilot_train_infos.read_bytes()
    result = select(pickle.loads(data), args.per_class)
    result["infos_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
