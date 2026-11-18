#!/usr/bin/env bash
# Multi-GPU training on one node: bash tools/dist_train.sh CONFIG GPUS [train_policy.py args]
# The global batch size is schedule.batch_size (split evenly over the GPUs).
set -euo pipefail
CONFIG=$1
GPUS=$2
shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT:$ROOT/third_party/openpi/src:$ROOT/third_party/openpi_client/src:${PYTHONPATH:-}"
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    "$ROOT/tools/train_policy.py" "$CONFIG" "$@"
