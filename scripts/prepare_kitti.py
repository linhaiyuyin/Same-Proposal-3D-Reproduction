"""Prepare train/val infos and the train-only object sampling database."""
import argparse
import pickle
from pathlib import Path

import yaml
from easydict import EasyDict
from pcdet.datasets.kitti.kitti_dataset import KittiDataset


def prepare(root, workers, config):
    dataset = KittiDataset(EasyDict(yaml.safe_load(config.read_text())),
                           ["Car", "Pedestrian", "Cyclist"], training=False, root_path=root)
    for split in ("train", "val"):
        dataset.set_split(split)
        records = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
        ids = (root / "ImageSets" / f"{split}.txt").read_text().splitlines()
        if [r["point_cloud"]["lidar_idx"] for r in records] != ids:
            raise ValueError(f"split mismatch: {split}")
        output = root / f"kitti_infos_{split}.pkl"
        temporary = output.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(records, handle)
        temporary.replace(output)
    dataset.set_split("train")
    dataset.create_groundtruth_database(root / "kitti_infos_train.pkl", split="train")


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "storage/datasets/kitti")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.root, args.workers, project / "CLEAN-main/tools/cfgs/dataset_configs/kitti_dataset.yaml")
