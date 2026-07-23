#!/usr/bin/env bash
# 用法: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_fusion.sh
MMSEG_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
    GPUS=${#GPU_ARR[@]}
else
    CUDA_VISIBLE_DEVICES=0,1,2,3
    GPUS=4
fi
export CUDA_VISIBLE_DEVICES

cd "$MMSEG_ROOT"
export PYTHONPATH="$MMSEG_ROOT:${PYTHONPATH:-}"

# mmengine 分布式验证在 NFS 上创建 .dist_test/ 临时目录时存在竞态条件。
# 将 TMPDIR 指向本地磁盘（非 NFS），避免 OSError: Directory not empty。
export TMPDIR="/tmp/mmseg_${USER}_$$"
mkdir -p "$TMPDIR"

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=$GPUS --master_port=${PORT:-29553} \
    train.py configs/fusion_vits_m2f_nocont_23999.py --launcher pytorch \
    --work-dir work_dirs/fusion_vits_m2f_nocont_23999 "$@"
