#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_msdge_cifar100"

python3 ../eval_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch msdnet_ge \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --batch-size 128 \
    --nBlocks 10 \
    --stepmode even \
    --step 2 \
    --base 4 \
    --nChannels 16 \
    --evalmode anytime \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --workers 1 \
    --gpu 0