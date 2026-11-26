"""Build only the deployed detector and require all inference tensors to load."""
import os

import torch
from pcdet.config import cfg
from pcdet.models import build_network
from pcdet.utils.pruning_utils import prune_plain_model


def numerical_protocol():
    return "strict-fp32-v1" if os.environ.get("W3219_STRICT_FP32") == "1" else "original"


def load_detector(dataset, weights):
    if numerical_protocol() == "strict-fp32-v1":
        if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("launch strict evaluation with scripts/project_fp32.sh")
        torch.manual_seed(666)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.MODEL.pop("KD_TRANS_BLOCK", None)
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    if cfg.get("PRUNING_RATE", 1.0) < 1:
        cfg.INHERIT = False
        model, _ = prune_plain_model.pruning(cfg, model, dataset)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    _, loaded = model._load_state_dict(checkpoint["model_state"], strict=True)
    if set(loaded) != set(model.state_dict()):
        raise ValueError("incomplete deployment checkpoint")
    progress = dict(checkpoint_epoch=int(checkpoint["epoch"]),
                    optimizer_updates=int(checkpoint["it"]) if "it" in checkpoint else None)
    return model.cuda().eval(), progress
