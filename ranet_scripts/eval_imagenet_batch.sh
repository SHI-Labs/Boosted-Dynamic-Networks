#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_ranet_imagenet_batch_model1"

# model 1: --nChannels 32 --grFactor 4-2-2-1 --bnFactor 4-2-2-1 \
# model 2: --nChannels 64 --grFactor 4-2-2-1 --bnFactor 4-2-2-1 \
python3 ../eval_imagenet.py \
    --data-root ${curr_dir}/../data/imagenet \
    --dataset imagenet \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch ranet \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --flat_curve \
    --batch-size 64 \
    --growthRate 16 \
    --stepmode even \
    --nBlocks 2 \
    --step 8 \
    --scale-list 1-2-3-4 \
    --nChannels 32 --grFactor 4-2-2-1 --bnFactor 4-2-2-1 \
    --evalmode dynamic \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --use-valid \
    --val_workers 16 \
    --gpu 0