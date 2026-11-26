"""Select channels by a candidate criterion within each original CLEAN layer budget."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def select(table, score, author, rate, output):
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["index"]) for row in rows] != list(range(3456)):
        raise ValueError("missing or reordered source channels")
    candidate = np.array([float(row[score]) for row in rows])
    published = torch.as_tensor(np.loadtxt(author), device="cuda")
    if published.ndim == 2:
        published = published.sum(0)
    if published.numel() != len(rows) or not np.isfinite(candidate).all():
        raise ValueError("invalid scores")
    count = int(rate * len(rows))
    top = published.sort(descending=True).indices[:count].cpu().numpy()
    global_top = np.argsort(-candidate, kind="stable")[:count]
    by_layer = {}
    for i, row in enumerate(rows):
        by_layer.setdefault(row["parameter"], []).append(i)
    selected, report = [], []
    for layer, ids in by_layer.items():
        ids = np.array(ids)
        quota = max(1, int(np.isin(ids, top).sum()))
        rank = ids[np.argsort(-candidate[ids], kind="stable")]
        selected.extend(rank[:quota])
        report.append(dict(layer=layer, full_width=len(ids), retained=quota,
                           global_candidate_width=max(1, int(np.isin(ids, global_top).sum())),
                           retained_local_indices=(rank[:quota] - ids[0]).tolist()))
    if len(selected) != count:
        raise ValueError("author minimum-layer fallback changed total channel budget")
    selected = np.array(selected)
    # Distinct scores reproduce the registered mask even if the empirical scores contain ties.
    priorities = np.arange(len(rows), dtype=float) / len(rows)
    priorities[selected] = 2 + np.argsort(np.argsort(candidate[selected], kind="stable"), kind="stable")
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output / "measure.txt", priorities[None, :], fmt="%.12g")
    meta = dict(rate=rate, score=score, kept=len(selected),
                overlap_with_author_global=int(np.isin(selected, top).sum()), layers=report,
                source_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                author_sha256=hashlib.sha256(author.read_bytes()).hexdigest())
    (output / "mask.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", type=Path)
    p.add_argument("score")
    p.add_argument("output", type=Path)
    p.add_argument("--rate", type=float, default=0.4)
    p.add_argument("--author", type=Path, default=Path(__file__).resolve().parents[1] /
                   "CLEAN-main/tools/measure_doc/SECONDNet_kitti_mask.txt")
    a = p.parse_args()
    select(a.table, a.score, a.author, a.rate, a.output)
