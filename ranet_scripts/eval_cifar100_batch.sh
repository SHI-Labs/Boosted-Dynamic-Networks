#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_ranet_cifar100_batch_model1"

# model 1: --scale-list 1-2-3 --grFactor 4-2-1 --bnFactor 4-2-1 \
# model 2: --scale-list 1-2-2-3 --grFactor 4-2-2-1 --bnFactor 4-2-2-1 \
# model 3: --scale-list 1-2-3-3 --grFactor 4-2-1-1 --bnFactor 4-2-1-1 \
python3 ../eval_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir "${curr_dir}/../results/boostnet/$train_id" \
    --arch ranet \
    --save_suffix "_use_reweight" --ensemble_reweight 0.5 \
    --flat_curve \
    --batch-size 128 \
    --nBlocks 2 \
    --stepmode lin_grow \
    --step 2 \
    --nChannels 16 \
    --scale-list 1-2-3 --grFactor 4-2-1 --bnFactor 4-2-1 \
    --evalmode dynamic \
    --use-valid \
    --evaluate-from ${curr_dir}/../results/boostnet/$train_id/model_best.pth \
    --workers 1 \
    --gpu 0