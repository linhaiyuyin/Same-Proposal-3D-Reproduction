# Same-Proposal Box Distillation for Channel-Pruned LiDAR Detectors

This repository reproduces our LiDAR-only 3D detection experiments on KITTI. We train a pruned SECOND student with CLEAN and add a box loss from a frozen three-class Voxel R-CNN teacher. The student's training-time NMS selects anchors, and the teacher refines its boxes at those same anchors. We supervise the student's geometry with oriented 3D IoU against the refined teacher boxes. The deployed student keeps the original CLEAN architecture.

The release includes early-box, matched ground-truth and count-matched late-box controls. The count-matched control draws the same number of teacher boxes per frame and class as the quality gate on the current batch and model state; total supervision can differ after training trajectories diverge. A separate study reconstructs the first bird's-eye-view (BEV) convolution under the original CLEAN mask and compares it with aligned inheritance and batch-normalization calibration.

## KITTI validation results

| Setting | mAP | Paired Δ vs CLEAN | Parameters | Model P50 |
| --- | ---: | ---: | ---: | ---: |
| CLEAN ×0.40 | 72.64 | — | 1,312,349 | 9.34 ms |
| Quality-gated late box ×0.40 | 73.56 | +0.92 ± 0.09 | 1,312,349 | 9.38 ms |
| Count-matched late box ×0.40 | **73.79** | **+1.15 ± 0.13** | 1,312,349 | 9.17 ms |
| CLEAN ×0.25 | 69.47 | — | 604,664 | 8.33 ms |
| Quality-gated late box ×0.25 | 69.93 | +0.46 ± 0.79 | 604,664 | 8.37 ms |

mAP averages Moderate 3D AP@R40 for Car, Pedestrian and Cyclist at IoUs of 0.70, 0.50 and 0.50. Paired differences use seeds 666, 667 and 668 with sample standard deviation. The gated and count-matched ×0.40 recipes each improved all three seeds. The ×0.25 recipe improved two seeds. BEV reconstruction gained +0.32 ± 0.44 points over the BN control and +0.09 ± 0.31 over CLEAN, below the descriptive +0.50-point criterion. Batch-1 FP32 timing uses an RTX PRO 6000 Blackwell Server Edition.

Run `bash reproduce.sh table` with Python 3.10+ to rebuild the results table from the run-level CSVs. It includes per-class AP, both adapted baselines, the target ablations and the independent BEV study. The CSVs contain aggregate metrics and timing percentiles.

## Install

Use Linux x86-64, Python 3.10, Git, a C++ compiler supported by CUDA 13.0, and an NVIDIA driver compatible with your GPU and CUDA toolkit. The reference build used PyTorch 2.10.0, CUDA 13.0.88 and spconv-cu120 2.3.6. Run the commands below from this repository root. [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 setuptools==79.0.1 wheel==0.45.1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-build-isolation -r requirements.txt
python scripts/bootstrap.py
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=12.0
bash reproduce.sh build
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh smoke
```

Set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` for your local toolkit and GPU. The installation is GPU-specific; use a fresh checkout when updating the overlay. The example targets an RTX PRO 6000. Bootstrap fetches a pinned CLEAN commit and applies the files in `overlay/`. Pass `--upstream /path/to/CLEAN` to use an existing Git checkout containing that commit, or `--storage /path/to/new-run-storage` to place working files on a separate volume.

## Prepare KITTI and the starting weights

1. Download and extract the KITTI 3D object detection training files. The KITTI root must contain `training/velodyne`, `training/calib`, `training/image_2` and `training/label_2`. Obtain the road-plane archive used for ground-truth sampling. Calibration, labels, point clouds and image dimensions are used during preprocessing. The detector takes points as input. [KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) · [OpenPCDet road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view)
2. Get `kitti/second/checkpoint_epoch_80.pth` from the CLEAN authors' pretrained-model folder. The conversion command checks its SHA-256 and writes a tensor-only checkpoint. Only use the documented trusted original checkpoint: legacy PyTorch deserialization and local KITTI prediction pickles must not be used with untrusted inputs. SHA-256 values for the input archives and the paper checkpoints are in `results/`. [CLEAN pretrained models](https://drive.google.com/drive/folders/1VTSrXW8MiW_1kbxZEPEMTxq8ZIXUVzuw?usp=sharing)

```bash
python scripts/prepare_inputs.py kitti /path/to/KITTI --planes-zip /path/to/train_planes.zip
bash reproduce.sh prepare
bash reproduce.sh second /path/to/kitti/second/checkpoint_epoch_80.pth
```

Preparation verifies the 3,712/3,769 frame split and builds the training object database and evaluation metadata under `storage/`. If your KITTI root already contains the matching `training/planes` directory, registration can use it directly.

### Shared teacher

Train the three-class teacher with seed 666. Every student run uses the same teacher file. The teacher used for the paper is identified by its SHA-256 in `results/inputs.json`. The command below trains a teacher locally for a new student run. Evaluation of a saved student requires only its checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh train teacher 666
teacher_run=$(bash reproduce.sh run-dir teacher 666)
ln -s "$(realpath "$teacher_run/ckpt/checkpoint_epoch_80.pth")" \
  storage/checkpoints/kitti/voxel-rcnn/checkpoint_epoch_80.pth
```

## Train and evaluate

The paper uses two GPUs and a global batch of 16 for 80 epochs and 18,560 student optimizer updates. Each recipe uses the epoch-80 checkpoint. Rerunning the training command resumes the latest verified local checkpoint. Training logs and outputs are under `storage/outputs/`.

```bash
# One real teacher/student training step with loss and gradient checks
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh smoke-kd count-matched040 storage/kd-smoke.json

# Reproduce the primary baseline and late-box results
for seed in 666 667 668; do
  for recipe in clean040 late040 count-matched040; do
    CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh train "$recipe" "$seed"
    run=$(bash reproduce.sh run-dir "$recipe" "$seed")
    ckpt="$run/ckpt/checkpoint_epoch_80.pth"
    evaluation="$run/eval/kitti-val"
    CUDA_VISIBLE_DEVICES=0 bash reproduce.sh eval "$recipe" "$ckpt" "$evaluation"
    CUDA_VISIBLE_DEVICES=0 bash reproduce.sh time "$recipe" "$ckpt" "$evaluation/timing.json"
  done
done
bash reproduce.sh table --runs storage/outputs
```

`eval` accepts any compatible checkpoint and a new output directory. It records predictions, per-class metrics, the checkpoint hash and deployed parameters. `time` uses 100 warmup frames and 500 timed frames in each of three repeats. Model time includes decode and NMS. End-to-end time adds input preparation and device transfer. Use `table --runs` with the output layout above to summarize completed three-seed groups.

| Study | Recipes |
| --- | --- |
| Target ablations at ×0.40 | `early040`, `matched-gt040`, `count-matched040` |
| Local matched-width baseline adaptations | `fisher040`, `sparsekd040` |
| Second retention setting | `clean025`, `late025` |
| BEV initialization | `clean040`, `bev-bn`, `bev-reconstructed` |
| Full SECOND and pruning without distillation | `second`, `no-kd040` |

The Fisher recipe uses the layer-matched ranking vector included in the release. Run `CUDA_VISIBLE_DEVICES=0 bash reproduce.sh fit-fisher` to recompute it from the 96 training frames. The command prints the path to the new vector and preserves the bundled reference. [Recipe index](recipes.tsv)

### BEV initialization and overlap analysis

Run the calibration once before training `bev-bn` and `bev-reconstructed`. Both recipes use the fixed ×0.40 mask, the original CLEAN distillation and the same BN calibration procedure. The ridge residual uses 66 training frames and reports reconstruction diagnostics on another 30 training frames.

```bash
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh fit-bev
CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh train bev-bn 666
CUDA_VISIBLE_DEVICES=0,1 bash reproduce.sh train bev-reconstructed 666
```

Use `cyclist-overlap` with saved validation predictions to export AP and precision curves at IoU 0.50, 0.60 and 0.70. The `fit-bev` report gives the held-out BEV localization coverage and head output errors.

```bash
run=$(bash reproduce.sh run-dir late040 666)
CUDA_VISIBLE_DEVICES=0 bash reproduce.sh cyclist-overlap \
  "$run/eval/kitti-val/result.pkl" storage/cyclist-overlap.csv
```

## CPU-only core check

The following optional check exercises the shipped oriented-IoU implementation and the stage-target logic with synthetic boxes. It uses a CPU replacement for the CUDA matching kernel and independently compares IoU with Shapely. It does not validate CUDA operators, the complete detector, or KITTI retraining.

```bash
python -m pip install shapely
python scripts/smoke_stage_cpu.py
```

## Source and artifact availability

The release contains source code, configuration, and run-level result summaries. KITTI, derived per-frame data, model weights, raw predictions and binaries are acquired or generated separately. The supplementary pilot results are reported in the manuscript. Exact numerical replay requires the original trained checkpoints, identified by SHA-256 in `results/checkpoints.csv`; they are not currently included or linked. Retraining from the recipe can yield different weights and metrics. [Late-box loss](overlay/pcdet/models/kd_heads/anchor_head/stage_target_distillation.py) · [BEV initialization](scripts/fit_bev.py)

The experiment team reports replaying CLEAN, count-matched late box and BEV reconstruction at seed 666 on all 3,769 validation frames from a clean release checkout. The public package includes summaries and hashes, not the underlying replay logs or checkpoints. Preprocessing, both BEV initialization patches and the Fisher ranking were regenerated and checked against the archived hashes. Use `sha256sum -c SHA256SUMS` to check the release files.

CLEAN is pinned at `0b4f431d11809c1f1e7c2b43618dc8ba64123405`. `overlay-sources.json` records file hashes from the frozen experiments and the portable release. The modified CLEAN sources retain their original notices. [CLEAN, TPAMI 2025](https://doi.org/10.1109/TPAMI.2025.3582706) · [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) · [SparseKD](https://github.com/CVMI-Lab/SparseKD)

Source is released under [Apache-2.0](LICENSE). The bundled KITTI evaluation modifications retain the [upstream MIT notice](overlay/pcdet/datasets/kitti/kitti_object_eval_python/LICENSE).
