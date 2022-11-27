#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_ranet_imagenet_any"

python3 ../eval_imagenet.py \
    --data-root ${curr_dir}/../data/imagenet \
    --dataset imagenet \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch ranet \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --batch-size 64 \
    --growthRate 16 \
    --nChannels 64 \
    --stepmode even \
    --nBlocks 2 \
    --step 8 \
    --scale-list 1-2-3-4 \
    --grFactor 4-2-2-1 \
    --bnFactor 4-2-2-1 \
    --evalmode anytime \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --val_workers 16 \
    --gpu 0