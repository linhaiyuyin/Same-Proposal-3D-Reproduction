"""Derive a deterministic pilot split and matching augmentation pool from train only."""
import argparse
import hashlib
import pickle
from pathlib import Path


def write_pickle(filename, value):
    temporary = filename.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    temporary.replace(filename)


def split(root):
    ids = (root / "ImageSets/train.txt").read_text().splitlines()
    val = set((root / "ImageSets/val.txt").read_text().splitlines())
    if len(ids) != 3712 or len(set(ids)) != len(ids) or val.intersection(ids):
        raise ValueError("expected the original disjoint KITTI train and val splits")
    key = lambda frame: hashlib.sha256(("w3219-v1/" + frame).encode()).digest()
    dev = set(sorted(ids, key=key)[:512])
    pilot_train = set(ids) - dev
    with (root / "kitti_infos_train.pkl").open("rb") as stream:
        infos = pickle.load(stream)
    if [v["point_cloud"]["lidar_idx"] for v in infos] != ids:
        raise ValueError("infos out of order")
    for name, frames in (("pilot_train", pilot_train), ("pilot_val", dev)):
        records = [i for i in infos if i["point_cloud"]["lidar_idx"] in frames]
        (root / "ImageSets" / f"{name}.txt").write_text(
            "".join(i["point_cloud"]["lidar_idx"] + "\n" for i in records))
        write_pickle(root / f"kitti_infos_{name}.pkl", records)
    with (root / "kitti_dbinfos_train.pkl").open("rb") as stream:
        database = pickle.load(stream)
    subset = {cls: [obj for obj in objects if obj["image_idx"] in pilot_train]
              for cls, objects in database.items()}
    write_pickle(root / "kitti_dbinfos_pilot_train.pkl", subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    split(parser.parse_args().root)
