#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id='exp0_msdge_cifar100_batch'

python3 ../eval_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --arch msdnet_ge \
    --batch-size 128 \
    --nBlocks 6 \
    --stepmode even \
    --step 2 \
    --base 4 \
    --nChannels 16 \
    --flat_curve \
    --evalmode dynamic \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --use-valid \
    --workers 1 \
    --gpu 0