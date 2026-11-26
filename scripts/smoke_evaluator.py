"""Check KITTI rotated-overlap geometry and R40 on a known perfect detector."""
import copy

import numpy as np
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval


def main():
    boxes = np.array([[2, 3, 4, 1, 0], [2, 3, 4, 1, np.pi / 2],
                      [8, 8, 1, 1, 0]], dtype=np.float32)
    expected = [[1, 1 / 7, 0], [1 / 7, 1, 0], [0, 0, 1]]
    np.testing.assert_allclose(rotate_iou_gpu_eval(boxes, boxes), expected, atol=1e-5)
    ann = dict(name=np.array(["Car"]), truncated=np.zeros(1), occluded=np.zeros(1),
               alpha=np.zeros(1), bbox=np.array([[100., 100., 200., 200.]]),
               dimensions=np.array([[4., 1.5, 1.6]]), location=np.array([[0., 1., 20.]]),
               rotation_y=np.zeros(1), score=np.ones(1))
    gt = [copy.deepcopy(ann) for _ in range(64)]
    pred = copy.deepcopy(gt)
    _, metrics = get_official_eval_result(gt, pred, ["Car"])
    np.testing.assert_allclose(metrics["Car_3d/moderate_R40"], 100.)
    print("PASS rotated overlap and perfect KITTI AP_R40")


if __name__ == "__main__":
    main()
