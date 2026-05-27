#!/bin/sh
set -e

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

"$PYTHON" src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --exp_name cheetah --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
  --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 \
  --exp_name cheetah_baseline --wandb_mode online --video_log_freq 10
