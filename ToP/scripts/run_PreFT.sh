#!/bin/bash

# Example run: bash run_PreFT.sh [TASK] [EX_NAME_SUFFIX] [GPU_ID]

glue_low=(MRPC RTE STSB CoLA WNLI)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}


# task and data
task_name=$1

# pretrain model
model_name_or_path=bert-base-uncased

# logging & saving
logging_steps=10
save_steps=0
if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=10
	epochs=10
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
	epochs=3
fi

# train parameters
max_seq_length=256
batch_size=32
learning_rate=2e-5

# seed
seed=57

# output directory
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix}
output_dir=out/${task_name}/${ex_name}
mkdir -p $output_dir
pruning_type=None

CUDA_VISIBLE_DEVICES=$3 python3 $code_dir/run_glue_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} 2>&1 | tee ${output_dir}/log.txt

