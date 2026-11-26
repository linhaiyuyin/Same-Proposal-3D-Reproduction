"""Exercise a candidate's actual teacher, augmented GT association and training gradient."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, build_teacher_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.kd_utils import kd_utils, kd_forwad
from pcdet.utils.pruning_utils.prune_plain_model import pruning


def smoke(config, output):
    project = Path(__file__).resolve().parents[1]
    os.chdir(project / "CLEAN-main/tools")
    cfg_from_yaml_file(str(config),cfg)
    kd_utils.process_kd_config()
    common_utils.set_random_seed(665)
    logger=common_utils.create_logger()
    ds=KittiDataset(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=True,logger=logger)
    full=build_network(cfg.MODEL,3,ds)
    full._load_state_dict(torch.load(cfg.PRETRAINED_MODEL,map_location='cpu',weights_only=True)['model_state'],strict=False)
    model,_=pruning(cfg,full,ds)
    model.cuda().train()
    teacher=build_teacher_network(cfg,SimpleNamespace(teacher_ckpt=cfg.TEACHER_CKPT),ds,False,logger).eval()
    kd_utils.prepare_kd_modules(teacher,model)
    batch=ds.collate_batch([ds[i] for i in (0,6,17,22)])
    optimizer=torch.optim.SGD(model.parameters(),lr=0.0003)
    loss,metrics,_=kd_forwad.forward(model,teacher,batch,optimizer,None,cfg.OPTIMIZATION,load_data_to_gpu)
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('invalid training step')
    if 'stage_pairs' in metrics:
        if metrics['stage_pairs'] <= 0 or not 0 <= metrics['stage_replaced'] <= metrics['stage_pairs']:
            raise ValueError(f'invalid paired stage targets: {metrics}')
    elif metrics['kd_gt_matched']<=0 or abs(metrics['kd_valid']-metrics['kd_weight_sum'])>0.01:
        raise ValueError(f'no matched real objects or unequal KD budget: {metrics}')
    output.write_text(json.dumps(dict(loss=loss.item(),metrics=metrics),indent=2,allow_nan=False)+'\n')


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    smoke(args.config.resolve(),args.output.resolve())
