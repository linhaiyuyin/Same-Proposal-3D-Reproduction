"""Print per-method KITTI results with paired differences and sample SD."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import sys

from bootstrap import ROOT

CLASSES = ("Car", "Pedestrian", "Cyclist")
SEEDS = (666, 667, 668)
COHORTS = ("0.40", "0.25", "BEV 0.40")
FIELDS = ("model_p50_ms", "model_p95_ms", "e2e_p50_ms", "e2e_p95_ms")


def archived_rows():
    for name in ("seed_results.csv", "bev_seeds.csv"):
        with (ROOT / "results" / name).open(newline="") as stream:
            for row in csv.DictReader(stream):
                yield row.get("retention", "BEV 0.40"), row["method"], int(row["seed"]), row


def load(path):
    return json.loads(path.read_text())


def measured_row(path, cohort):
    metrics = load(path / "metrics.json")
    evidence = load(path / "inference.json")
    expected = 604664 if cohort == "0.25" else 1312349
    if (evidence["numerical_protocol"], evidence["frames"], metrics["frames"],
            evidence["batch"], evidence["checkpoint_epoch"], evidence["optimizer_updates"],
            evidence["deployed_parameters"]) != ("strict-fp32-v1", 3769, 3769, 4, 80, 18560, expected):
        raise ValueError(f"incorrect inference or training protocol in {path}")
    row = {name: metrics["metrics"][f"{name}_3d/moderate_R40"] for name in CLASSES}
    row.update(mAP=metrics["moderate_3d_map_r40"], params=expected)
    timing = path / "timing.json"
    if timing.is_file():
        info = load(timing)
        if (info["numerical_protocol"], info["warmup"], len(info["frames"]), len(info["runs"]),
                info["weights_sha256"], info["parameters"]) != (
                "strict-fp32-v1", 100, 500, 3, evidence["weights_sha256"], expected):
            raise ValueError(f"incorrect or mismatched timing protocol in {timing}")
        row.update(zip(FIELDS, [info[k][p] for k in ("pooled", "pooled_end_to_end")
                                for p in ("p50_ms", "p95_ms")]))
    return row


def measured_rows(outputs):
    with (ROOT / "recipes.tsv").open(newline="") as stream:
        recipes = list(csv.DictReader(stream, delimiter="\t"))
    indexed = {(row["cohort"], row["method"]): row["config"] for row in recipes if row["cohort"]}
    indexed["BEV 0.40", "CLEAN"] = indexed["0.40", "CLEAN"]
    for (cohort, method), config in indexed.items():
        group = config.removeprefix("cfgs/").removesuffix(".yaml")
        paths = [outputs / group / f"s{seed}/eval/kitti-val" for seed in SEEDS]
        present = [(path / "metrics.json").is_file() and (path / "inference.json").is_file()
                   for path in paths]
        if not all(present):
            if any(present):
                print(f"Skipping {cohort} {method}: all three paired seeds are required", file=sys.stderr)
            continue
        for seed, path in zip(SEEDS, paths):
            yield cohort, method, seed, measured_row(path, cohort)


def paired(a, b):
    changes = [float(a[seed]["mAP"]) - float(b[seed]["mAP"]) for seed in SEEDS]
    return f"{fmean(changes):+.2f} ± {stdev(changes):.2f} ({sum(v > 0 for v in changes)}/3)"


def report(rows):
    groups = defaultdict(dict)
    for cohort, method, seed, row in rows:
        mean = fmean(float(row[name]) for name in CLASSES)
        if not math.isclose(mean, float(row["mAP"]), abs_tol=1e-9) or seed in groups[cohort, method]:
            raise ValueError(f"invalid {cohort} {method} seed {seed} result")
        groups[cohort, method][seed] = row
    if not groups:
        raise ValueError("no complete paired result sets")
    print("| Width / study | Method | Car | Pedestrian | Cyclist | mAP | Δ vs CLEAN (improved) | Parameters | Model P50/P95 | End-to-end P50/P95 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cohort in COHORTS:
        baseline = groups.get((cohort, "CLEAN"))
        for (study, method), runs in groups.items():
            if study != cohort:
                continue
            params = {int(row["params"]) for row in runs.values()}
            if tuple(sorted(runs)) != SEEDS or len(params) != 1:
                raise ValueError(f"invalid seed or parameter set for {study} {method}")

            def mean(field):
                return f"{fmean(float(row[field]) for row in runs.values()):.2f}"

            def times(fields):
                return " / ".join(map(mean, fields)) if all(
                    all(field in row for field in fields) for row in runs.values()) else "—"

            delta = paired(runs, baseline) if baseline and method != "CLEAN" else "—"
            cells = [study, method, *(mean(name) for name in CLASSES), mean("mAP"), delta,
                     f"{params.pop():,}", times(FIELDS[:2]), times(FIELDS[2:])]
            print("| " + " | ".join(cells) + " |")
    bn = groups.get(("BEV 0.40", "BEV BN calibration"))
    reconstruction = groups.get(("BEV 0.40", "BEV reconstruction"))
    if bn and reconstruction:
        print(f"\nBEV reconstruction vs BN calibration {paired(reconstruction, bn)} AP points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, help="print complete groups from an alternative outputs directory")
    args = parser.parse_args()
    report(measured_rows(args.runs) if args.runs else archived_rows())
