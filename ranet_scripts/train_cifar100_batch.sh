#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
model="model1"
train_id="exp0_ranet_cifar100_batch_model1"
result_dir="${curr_dir}/../results/boostnet/$train_id"
mkdir -p $result_dir

if [[ "$model" == "model1" ]]; then
    scale_list="1-2-3"
    grFactor="4-2-1"
    bnFactor="4-2-1"
elif [[ "$model" == "model2" ]]; then
    scale_list="1-2-2-3"
    grFactor="4-2-2-1"
    bnFactor="4-2-2-1"
elif [[ "$model" == "model3" ]]; then
    scale_list="1-2-3-3"
    grFactor="4-2-1-1"
    bnFactor="4-2-1-1"
fi

python3 ../train_cifar100.py \
    --data-root ${curr_dir}/../data/cifar100 \
    --dataset cifar100 \
    --result_dir $result_dir \
    --use-valid \
    --arch ranet \
    --ensemble_reweight 0.5 \
    --batch-size 64 \
    --nBlocks 2 \
    --stepmode lin_grow \
    --step 2 \
    --scale-list $scale_list --grFactor $grFactor --bnFactor $bnFactor \
    --nChannels 16 \
    --workers 8 \
    --lr_f 0.1 \
    --lr_milestones '150,225' \
    --epochs 300 \
    --weight-decay 1e-4