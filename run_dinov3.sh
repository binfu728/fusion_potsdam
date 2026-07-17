#!/usr/bin/env bash
# 纯 DINOv3 frozen finetune: CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_dinov3.sh
MMSEG_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
    GPUS=${#GPU_ARR[@]}
else
    CUDA_VISIBLE_DEVICES=0,1,2,3; GPUS=4
fi
export CUDA_VISIBLE_DEVICES
cd "$MMSEG_ROOT"
export PYTHONPATH="$MMSEG_ROOT:${PYTHONPATH:-}"
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=$GPUS --master_port=${PORT:-29551} \
    train.py configs/dinov3_m2f.py --launcher pytorch \
    --work-dir work_dirs/dinov3_m2f "$@"
