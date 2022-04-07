#!/bin/bash

# Training
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
	     --class_cond True --no_instance True

# Classifier-free Finetune
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-FINETUNE'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
	     --class_cond True --no_instance True --drop_rate 0.2 --resume_checkpoint OUTPUT/ADE20K-SDM-256CH/model.pt

# Testing
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-TEST'
mpiexec -n 8 python image_sample.py --data_dir ./data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
       --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 151 \
       --class_cond True --no_instance True --batch_size 2 --num_samples 2000 --model_path OUTPUT/ADE20K-SDM-256CH-FINETUNE/ema_0.9999_best.pt --results_path RESULTS/ADE20K-SDM-256CH --s 1.5

