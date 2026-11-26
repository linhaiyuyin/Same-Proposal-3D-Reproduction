"""Channel order used by the original SECOND pruning score."""


def parameters(model):
    offset = 0
    for name, tensor in model.named_parameters():
        if name.startswith("backbone_3d.") and tensor.ndim == 5:
            axis = 0
        elif name.startswith("backbone_2d.blocks.") and tensor.ndim == 4:
            axis = 0
        elif name.startswith("backbone_2d.deblocks.") and tensor.ndim == 4:
            axis = 1
        else:
            continue
        width = tensor.shape[axis]
        yield name, tensor, axis, offset, width
        offset += width
    if offset != 3456:
        raise ValueError(f"unexpected SECOND width total {offset}")

