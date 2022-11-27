#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_ranet_cifar100_any"
result_dir="${curr_dir}/../results/boostnet/$train_id"
mkdir -p $result_dir

python3 ../train_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir $result_dir \
    --arch ranet \
    --ensemble_reweight 0.5 \
    --batch-size 64 \
    --nBlocks 2 \
    --stepmode even \
    --step 4 \
    --nChannels 16 \
    --scale-list 1-2-3-3 \
    --grFactor 4-2-1-1 \
    --bnFactor 4-2-1-1 \
    --workers 16 \
    --lr_f 0.1 \
    --lr_milestones '150,225' \
    --epochs 300 \
    --weight-decay 1e-4