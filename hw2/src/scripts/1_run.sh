#!/bin/sh
set -e

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  --exp_name cartpole --wandb_mode online --video_log_freq 10
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg --exp_name cartpole_rtg --wandb_mode online --video_log_freq 10
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -na --exp_name cartpole_na --wandb_mode online --video_log_freq 10
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg -na --exp_name cartpole_rtg_na --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  --exp_name cartpole_lb --wandb_mode online --video_log_freq 400
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -rtg --exp_name cartpole_lb_rtg --wandb_mode online --video_log_freq 10
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -na --exp_name cartpole_lb_na --wandb_mode online --video_log_freq 10
"$PYTHON" src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
  -rtg -na --exp_name cartpole_lb_rtg_na --wandb_mode online --video_log_freq 10
