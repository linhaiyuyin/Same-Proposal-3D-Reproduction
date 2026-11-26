"""Check CUDA execution, backward, and non-maximum suppression before training."""
import torch
import spconv.pytorch as spconv

from pcdet.ops.iou3d_nms import iou3d_nms_utils as box_ops
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda
from pcdet.ops.roipoint_pool3d import roipoint_pool3d_cuda


def main():
    boxes = torch.tensor([[0, 0, 0, 2, 2, 2, 0], [0, 0, 0, 2, 2, 2, 0],
                          [6, 0, 0, 2, 2, 2, 0]], dtype=torch.float32, device="cuda")
    expected = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1]], device="cuda").float()
    torch.testing.assert_close(box_ops.boxes_iou3d_gpu(boxes, boxes), expected)
    keep, _ = box_ops.nms_gpu(boxes, boxes.new_tensor([0.9, 0.7, 0.5]), 0.1)
    assert keep.tolist() == [0, 2]
    indices = torch.tensor([[0, 2, 2, 2], [0, 2, 2, 3]], device="cuda", dtype=torch.int32)
    features = torch.randn(2, 16, device="cuda", requires_grad=True)
    values = spconv.SparseConvTensor(features, indices, [6, 6, 6], 1)
    layer = spconv.SubMConv3d(16, 16, kernel_size=3, padding=1, bias=False).cuda()
    output = layer(values).features
    output.square().sum().backward()
    assert features.grad.isfinite().all() and features.grad.abs().sum() > 0
    print("PASS", torch.cuda.get_device_name(), torch.__version__, output.shape)


if __name__ == "__main__":
    main()
