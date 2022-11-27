#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_ranet_cifar100_any"

python3 ../eval_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch ranet \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --batch-size 128 \
    --nBlocks 2 \
    --stepmode even \
    --step 4 \
    --nChannels 16 \
    --scale-list 1-2-3-3 \
    --grFactor 4-2-1-1 \
    --bnFactor 4-2-1-1 \
    --evalmode anytime \
    --evaluate-from "${curr_dir}/../results/boostnet/$train_id/model_best.pth" \
    --val_workers 4 \
    --gpu 0