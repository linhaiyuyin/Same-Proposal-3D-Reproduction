#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=$(command -v "${PYTHON:-python3}")
clean="$root/CLEAN-main"
tools="$clean/tools"
work="$root/storage"
command=${1:-help}
if (( $# )); then shift; fi

config_path() {
    awk -F '\t' -v recipe="$1" '$1 == recipe {print $2; found=1; exit} END{exit !found}' "$root/recipes.tsv" ||
        { echo "Unknown recipe: $1" >&2; return 2; }
}

run_dir() {
    local config group
    [[ "$2" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer" >&2; return 2; }
    config=$(config_path "$1") || return
    group=${config#cfgs/}
    printf '%s\n' "$work/outputs/${group%.yaml}/s$2"
}

if [[ "$command" == help || "$command" == --help ]]; then
    cat <<EOF
Usage: ./reproduce.sh <command> [arguments]
  build
  smoke
  prepare [DATA_WORKERS]
  second SECOND_CHECKPOINT
  train RECIPE SEED
  run-dir RECIPE SEED
  eval RECIPE CHECKPOINT NEW_OUTPUT_DIR
  time RECIPE CHECKPOINT OUTPUT_JSON
  smoke-kd RECIPE OUTPUT_JSON
  fit-bev
  fit-fisher
  cyclist-overlap PREDICTIONS OUTPUT_CSV
  table [--runs OUTPUT_ROOT]
EOF
    exit 0
fi
if [[ "$command" == table ]]; then
    exec "$python_bin" "$root/scripts/summarize.py" "$@"
fi
if [[ "$command" == run-dir ]]; then
    run_dir "${1:?Expected a recipe name}" "${2:?Expected an integer seed}"
    exit
fi
if [[ ! -f "$clean/.clean-commit" ]]; then
    echo "Run python3 scripts/bootstrap.py first." >&2
    exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONPATH="$clean" NUMBA_CUDA_USE_NVIDIA_BINDING=1
if [[ "$command" == second ]]; then
    exec "$python_bin" "$root/scripts/prepare_inputs.py" second "$@"
fi
nvvm=$("$python_bin" -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("nvidia.cuda_nvcc").origin).parent / "nvvm/lib64")')
export LD_LIBRARY_PATH="$nvvm${CUDA_HOME:+:$CUDA_HOME/lib64:$CUDA_HOME/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

strict_fp32() {
    export W3219_STRICT_FP32=1 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4
}

case "$command" in
    build)
        : "${CUDA_HOME:?Set CUDA_HOME to the CUDA 13.0 toolkit prefix}"
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH" CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
        export MAX_JOBS="${MAX_JOBS:-4}"
        cd "$clean"
        exec "$python_bin" setup.py build_ext --inplace \
            --build-temp "$work/build/temp" --build-lib "$work/build/lib"
        ;;
    smoke)
        "$python_bin" "$root/scripts/smoke_ops.py"
        exec "$python_bin" "$root/scripts/smoke_evaluator.py"
        ;;
    prepare)
        export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
        data="$work/datasets/kitti"
        exec 9>"$data/.prepare.lock"
        flock 9
        "$python_bin" "$root/scripts/prepare_kitti.py" --root "$data" --workers "${1:-8}"
        "$python_bin" "$root/scripts/development_split.py" "$data"
        exec "$python_bin" "$root/scripts/calibration_split.py" \
            "$data/kitti_infos_pilot_train.pkl" "$work/chapter3-cal96.json"
        ;;
    train)
        recipe=${1:?Expected a recipe name}
        seed=${2:?Expected an integer seed}
        run=$(run_dir "$recipe" "$seed")
        config=$(config_path "$recipe")
        trainer=train_with_pruning.py
        if [[ "$recipe" == teacher || "$recipe" == second ]]; then trainer=train.py; fi
        mkdir -p "$run"
        exec 9>"$run/.train.lock"
        flock 9
        checkpoint=$("$python_bin" "$root/scripts/select_checkpoint.py" "$run/ckpt")
        if [[ "${checkpoint##*/}" == checkpoint_epoch_80.pth ]]; then
            printf "%s\n" "$checkpoint"
            exit 0
        fi
        resume=()
        if [[ -n "$checkpoint" ]]; then resume=(--ckpt "$checkpoint"); fi
        launch=$(mktemp -d "$run/launch-XXXXXXXX")
        cd "$tools"
        "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=2 "$trainer" \
            --cfg_file "$config" --launcher pytorch --seed "$seed" --extra_tag "s$seed" \
            --batch_size 16 --workers 4 --max_ckpt_save_num 4 "${resume[@]}" 2>&1 | tee "$launch/train.log"
        printf "%s\n" "$run/ckpt/checkpoint_epoch_80.pth"
        ;;
    eval)
        config=$(config_path "${1:?Expected a recipe name}")
        strict_fp32
        exec "$python_bin" "$root/scripts/predict_checkpoint.py" "$tools/$config" \
            "${2:?Expected a checkpoint}" "${3:?Expected a new output directory}"
        ;;
    time)
        config=$(config_path "${1:?Expected a recipe name}")
        strict_fp32
        exec "$python_bin" "$root/scripts/benchmark_student.py" --config "$tools/$config" \
            --weights "${2:?Expected a checkpoint}" --output "${3:?Expected an output JSON path}"
        ;;
    smoke-kd)
        config=$(config_path "${1:?Expected a late box recipe}")
        exec "$python_bin" "$root/scripts/smoke_object_kd.py" "$tools/$config" "${2:?Expected an output JSON path}"
        ;;
    fit-bev)
        strict_fp32
        exec 9>"$work/checkpoints/kitti/.fit-bev.lock"
        flock 9
        exec "$python_bin" "$root/scripts/fit_bev.py" "$work/chapter3-cal96.json" "$work/checkpoints/kitti/bev"
        ;;
    fit-fisher)
        output=$(mktemp -d "$work/outputs/fisher-fit-XXXXXXXX")
        "$python_bin" "$root/scripts/score_fisher_channels.py" "$work/chapter3-cal96.json" "$output/scored"
        "$python_bin" "$root/scripts/match_channel_budget.py" "$output/scored/channels.csv" Fisher "$output/mask"
        printf '%s\n' "$output/mask/measure.txt"
        ;;
    cyclist-overlap)
        strict_fp32
        exec "$python_bin" "$root/scripts/cyclist_overlap.py" "$work/datasets/kitti/kitti_infos_val.pkl" \
            "${1:?Expected a prediction pickle}" "${2:?Expected an output CSV path}"
        ;;
    *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
