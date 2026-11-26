"""Build the KITTI runtime from a pinned CLEAN commit and the release overlay."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/zhnxjtu/CLEAN.git"
COMMIT = "0b4f431d11809c1f1e7c2b43618dc8ba64123405"
SPLIT_SHA = {
    "train": "b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb",
    "val": "657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86",
}
MASK = "tools/measure_doc/SECONDNet_kitti_mask.txt"
MASK_SHA = "c72cbf2232a392686b970f1faae5af5a86f353c227924a03a67609ed6db695af"
UPSTREAM_FILES = (
    "LICENSE", "pcdet", "data/kitti/ImageSets/train.txt", "data/kitti/ImageSets/val.txt",
    "tools/_init_path.py", "tools/test.py", "tools/eval_utils", "tools/train_utils",
    "tools/cfgs/kitti_models/second.yaml", MASK,
)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def link(path, target):
    if path.exists() or path.is_symlink():
        if path.resolve() != target.resolve():
            raise FileExistsError(f"{path} already exists and differs from {target}")
        return
    path.symlink_to(target.resolve(), target_is_directory=True)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def extract(upstream, stage):
    archive = stage.parent / "upstream.tar"
    git(upstream, "archive", "--output", str(archive), COMMIT, *UPSTREAM_FILES)
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unexpected archive member {rel}")
            if rel.suffix == ".so":
                continue
            path = stage / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def bootstrap(upstream, storage):
    checkout = ROOT / "CLEAN-main"
    marker = checkout / ".clean-commit"
    if checkout.exists():
        if marker.is_file() and marker.read_text().strip() == COMMIT:
            if storage is not None:
                link(ROOT / "storage", storage)
            print(checkout)
            return
        raise FileExistsError(f"{checkout} is already in use")
    entry = ROOT / "storage"
    storage = (storage or entry).resolve()
    if storage != entry:
        link(entry, storage)
    for folder in ("datasets/kitti/ImageSets", "outputs", "checkpoints/kitti/second",
                   "checkpoints/kitti/voxel-rcnn"):
        (storage / folder).mkdir(parents=True, exist_ok=True)
    splits = storage / "datasets/kitti/ImageSets"
    with tempfile.TemporaryDirectory(prefix=".clean-", dir=ROOT) as scratch:
        workspace = Path(scratch)
        if upstream is None:
            upstream = workspace / "upstream"
            upstream.mkdir()
            git(upstream, "init", "--quiet")
            git(upstream, "fetch", "--quiet", "--depth=1", UPSTREAM, COMMIT)
        staged = workspace / "CLEAN-main"
        extract(upstream, staged)
        shutil.copytree(ROOT / "overlay", staged, dirs_exist_ok=True)

        for split, expected in SPLIT_SHA.items():
            source = staged / f"data/kitti/ImageSets/{split}.txt"
            destination = splits / source.name
            if digest(source) != expected:
                raise ValueError(f"upstream KITTI {split} split changed")
            if destination.exists() and digest(destination) != expected:
                raise ValueError(f"existing KITTI {split} split differs")
            if not destination.exists():
                shutil.copyfile(source, destination)
        if digest(staged / MASK) != MASK_SHA:
            raise ValueError("upstream CLEAN channel score file changed")

        shutil.rmtree(staged / "data")
        link(staged / "data", storage / "datasets")
        link(staged / "output", storage / "outputs")
        link(staged / "tools/pretrained_model", storage / "checkpoints")
        (staged / ".clean-commit").write_text(COMMIT + "\n")
        staged.rename(checkout)
    print(checkout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="existing Git repository containing the pinned commit")
    parser.add_argument("--storage", type=Path, help="external working directory")
    args = parser.parse_args()
    bootstrap(args.upstream, args.storage)
