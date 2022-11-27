#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_msdge_imagenet_batch"

python3 ${curr_dir}/../eval_imagenet.py \
    --data-root ${curr_dir}/../data/imagenet \
    --dataset imagenet \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch msdnet_ge \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --batch-size 1024 \
    --nBlocks 5 \
    --stepmode even \
    --step 7 \
    --base 7 \
    --nChannels 32 \
    --growthRate 16 \
    --grFactor 1-2-4-4 \
    --bnFactor 1-2-4-4 \
    --flat_curve \
    --evalmode dynamic \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --use-valid \
    --val_workers 16 \
    --gpu 0