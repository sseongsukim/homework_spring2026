#!/bin/sh
set -e

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

for discount in 1 0.99 0.95; do
  for n_layers in 2 3; do
    for layer_size in 64 128; do
      for batch_size in 100 1000 5000; do
        for learning_rate in 0.01 5e-3 3e-4; do
          for rtg in 0 1; do
            for normalize_advantages in 0 1; do
              for gae_lambda in 0 0.95 0.98 0.99 1; do
                rtg_flag=""
                rtg_name="nortg"
                if [ "$rtg" = "1" ]; then
                  rtg_flag="-rtg"
                  rtg_name="rtg"
                fi

                na_flag=""
                na_name="nona"
                if [ "$normalize_advantages" = "1" ]; then
                  na_flag="-na"
                  na_name="na"
                fi

                exp_name="pendulum_d${discount}_l${n_layers}_s${layer_size}_b${batch_size}_lr${learning_rate}_${rtg_name}_${na_name}_gae${gae_lambda}"

                "$PYTHON" src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b "$batch_size" -eb 1000 \
                  --discount "$discount" -l "$n_layers" -s "$layer_size" -lr "$learning_rate" \
                  $rtg_flag $na_flag --gae_lambda "$gae_lambda" \
                  --exp_name "$exp_name" --wandb_mode online --video_log_freq 10
              done
            done
          done
        done
      done
    done
  done
done
