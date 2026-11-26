"""Select a readable, finite local training checkpoint, starting at the newest epoch."""
import argparse
import pickle
import re
import sys
from pathlib import Path

import torch


def verify(path, epoch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != epoch or checkpoint["it"] <= 0:
        raise ValueError("inconsistent epoch or optimizer step")
    state = checkpoint["model_state"]
    optimizer = checkpoint["optimizer_state"]
    if not state or not optimizer["state"] or not optimizer["param_groups"]:
        raise ValueError("incomplete model or optimizer")
    tensors = list(state.values())
    for entry in optimizer["state"].values():
        tensors.extend(x for x in entry.values() if torch.is_tensor(x))
    if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in tensors):
        raise ValueError("non-finite model or optimizer state")


def select(directory):
    candidates = []
    for path in directory.glob("checkpoint_epoch_*.pth"):
        match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match[1]), path))
    for epoch, path in sorted(candidates, reverse=True):
        try:
            verify(path, epoch)
        except (OSError, EOFError, ValueError, RuntimeError, KeyError, TypeError,
                pickle.UnpicklingError) as error:
            print(f"Rejected {path}: {error}", file=sys.stderr)
            continue
        return path.resolve()
    if candidates:
        raise ValueError(f"No readable training checkpoint in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = select(args.directory)
    if result:
        print(result)
