#!/bin/sh
set -e

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

"$PYTHON" src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda 0 --exp_name lunar_lander_lambda0 --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95 --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98 --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99 --wandb_mode online --video_log_freq 10

"$PYTHON" src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 \
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline \
  --gae_lambda 1 --exp_name lunar_lander_lambda1 --wandb_mode online --video_log_freq 10
