"""Register external KITTI files and normalize the authors' SECOND checkpoint."""

import argparse
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

from bootstrap import ROOT, SPLIT_SHA, digest, link


SECOND_SHA = "05311f3815eeacf4966900d1cab342fb7770943751cf3bbf274c74817e9121f4"
PLANES_SHA = "672451441ebbbff360d2a9465f04712f0f4bb8fe033b70203485cf3d803cf2f4"


def register(raw, planes_zip):
    root = ROOT / "storage/datasets/kitti"
    if not (ROOT / "CLEAN-main/.clean-commit").is_file():
        raise FileNotFoundError("run scripts/bootstrap.py first")
    training = root / "training"
    training.mkdir(exist_ok=True)
    for folder in ("image_2", "label_2", "calib", "velodyne"):
        source = raw / "training" / folder
        if not source.is_dir():
            raise FileNotFoundError(source)
        link(training / folder, source)

    if planes_zip is not None:
        if digest(planes_zip) != PLANES_SHA:
            raise ValueError("unexpected OpenPCDet road-plane archive checksum")
        planes = training / "planes"
        if not planes.exists():
            with tempfile.TemporaryDirectory(prefix=".planes-", dir=training) as work:
                destination = Path(work) / "planes"
                destination.mkdir()
                with ZipFile(planes_zip) as archive:
                    files = [name for name in archive.namelist() if re.fullmatch(r"planes/[0-9]{6}\.txt", name)]
                    if len(files) != 7481 or len(set(files)) != len(files):
                        raise ValueError("unexpected road-plane file count")
                    for name in files:
                        (destination / Path(name).name).write_bytes(archive.read(name))
                destination.rename(planes)
    elif (raw / "training/planes").is_dir():
        link(training / "planes", raw / "training/planes")

    for split, checksum in SPLIT_SHA.items():
        split_file = root / f"ImageSets/{split}.txt"
        if digest(split_file) != checksum:
            raise ValueError(f"unexpected KITTI {split} split")
        for frame in split_file.read_text().splitlines():
            for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"),
                                   ("calib", ".txt"), ("velodyne", ".bin"), ("planes", ".txt")):
                path = training / folder / f"{frame}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)
    print("Registered KITTI training files and road planes")


def normalize_second(source):
    if digest(source) != SECOND_SHA:
        raise ValueError("expected the original SECOND epoch-80 checkpoint from CLEAN")
    import torch
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    if checkpoint["epoch"] != 80 or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("unexpected SECOND checkpoint metadata")
    path = ROOT / "storage/checkpoints/kitti/second/checkpoint_epoch_80.tensors.pth"
    with tempfile.TemporaryDirectory(prefix=".second-", dir=path.parent) as work:
        temporary = Path(work) / path.name
        torch.save(dict(model_state=state, epoch=80), temporary)
        temporary.replace(path)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    kitti = commands.add_parser("kitti")
    kitti.add_argument("root", type=Path)
    kitti.add_argument("--planes-zip", type=Path)
    second = commands.add_parser("second")
    second.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "kitti":
        register(args.root, args.planes_zip)
    else:
        normalize_second(args.checkpoint)
