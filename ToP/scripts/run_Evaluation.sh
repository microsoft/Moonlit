#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=$2

# logging & saving
save_steps=0


# train parameters
max_seq_length=256
batch_size=32

# seed
seed=57

# output dir
output_dir=/tmp/${task_name}/$$

pretrained_pruned_model=None


export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=$3 python3 $code_dir/run_glue_prune.py \
     --task ${task_name} \
	   --output_dir ${output_dir} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --overwrite_output_dir \
	   --seed ${seed} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --freeze_embeddings \
     --do_distill \
     --distillation_path $model_name_or_path \
     --use_mac_l0 \
     --dataloader_num_workers 0 \
     --log_level error \
     --eval_only
