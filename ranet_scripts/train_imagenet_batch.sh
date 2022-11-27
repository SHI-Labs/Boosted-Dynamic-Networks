#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
model="model1"
train_id="exp0_ranet_imagenet_batch_model1"
result_dir=${curr_dir}/../results/boostnet/$train_id
mkdir -p $result_dir

if [[ "$model" == "model1" ]]; then
    nChannels="32"
elif [[ "$model" == "model2" ]]; then
    nChannels="64"
fi

python3 -m torch.distributed.launch \
    --master_port=19999 \
    --nproc_per_node=4 \
    ${curr_dir}/../train_imagenet.py \
    --distributed \
    --data-root ${curr_dir}/../data/imagenet \
    --dataset imagenet \
    --result_dir $result_dir \
    --tensorboard_dir $result_dir/log \
    --use-valid \
    --arch ranet \
    --ensemble_reweight 0.5 \
    --epochs 90 \
    --batch-size 64 \
    --nBlocks 2 \
    --growthRate 16 \
    --stepmode even \
    --step 8 \
    --scale-list 1-2-3-4 \
    --nChannels $nChannels --grFactor 4-2-2-1 --bnFactor 4-2-2-1 \
    --workers 36 \
    --lr_f 0.1 \
    --lr_milestones '30,60' \
    --gpu 0,1,2,3 \
    --epochs 90 \
    --weight-decay 1e-4