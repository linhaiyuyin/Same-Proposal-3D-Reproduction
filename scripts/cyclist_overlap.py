"""Export the validation Cyclist precision curves at IoU 0.50, 0.60 and 0.70."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from pcdet.datasets.kitti.kitti_object_eval_python.eval import eval_class, get_mAP_R40


def evaluate(infos_path, prediction_path, output):
    with infos_path.open("rb") as stream:
        infos = pickle.load(stream)
    with prediction_path.open("rb") as stream:
        predictions = pickle.load(stream)
    frames = [row["point_cloud"]["lidar_idx"] for row in infos]
    if len(frames) != 3769 or len(set(frames)) != len(frames) or frames != [row["frame_id"] for row in predictions]:
        raise ValueError("expected predictions on the same 3,769 KITTI validation frames in order")

    thresholds = (.50, .60, .70)
    overlaps = np.broadcast_to(np.asarray(thresholds)[:, None, None], (len(thresholds), 3, 1)).copy()
    ground_truth = [row["annos"] for row in infos]
    precision = eval_class(ground_truth, predictions, [2], [1], 2, overlaps)["precision"]
    ap = get_mAP_R40(precision)[0, 0]
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("iou", "AP_R40", "nominal_recall", "interpolated_precision"))
        for index, threshold in enumerate(thresholds):
            for point, value in enumerate(precision[0, 0, index]):
                writer.writerow((threshold, float(ap[index]), point / 40, float(value)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infos", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.infos, args.predictions, args.output)
