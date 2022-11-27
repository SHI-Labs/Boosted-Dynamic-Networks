#!/bin/bash

curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
train_id="exp0_msd_imagenet_batch_step_5"
result_dir=${curr_dir}/../results/boostnet/$train_id
mkdir -p $result_dir

python3 -m torch.distributed.launch \
    --master_port=9999 \
    --nproc_per_node=4 \
    ${curr_dir}/../train_imagenet.py \
    --distributed \
    --data-root ${curr_dir}/../data/imagenet \
    --dataset imagenet \
    --result_dir $result_dir \
    --tensorboard_dir $result_dir/log \
    --use-valid \
    --arch msdnet_ge \
    --ensemble_reweight 0.5 \
    --batch-size 64 \
    --nBlocks 5 \
    --stepmode even \
    --step 5 \
    --base 5 \
    --nChannels 32 \
    --growthRate 16 \
    --grFactor 1-2-4-4 \
    --bnFactor 1-2-4-4 \
    --workers 36 \
    --lr_f 0.1 \
    --lr_milestones '30,60' \
    --gpu 0,1,2,3 \
    --epochs 90 \
    --weight-decay 1e-4