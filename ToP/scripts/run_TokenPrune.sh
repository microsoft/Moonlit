#!/bin/bash
glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=$6

# logging & saving
save_steps=0


# train parameters
max_seq_length=256
batch_size=32
learning_rate=${12}
reg_learning_rate=${13}

# seed
seed=57

# output dir
ex_name=$2
ex_cate=$3
output_dir=out/token_prune/${task_name}/${ex_cate}/${ex_name}

# pruning and distillation
pruning_type=$4
target_sparsity=$5
distillation_path=$6
distill_layer_loss_alpha=$7
distill_ce_loss_alpha=$8
distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=${9} 

# token pruning arguments
prune_location=${11}
bin_num=${15}
topk=${16}
epochs=${17}
eval_steps=${18}
scheduler_type=linear
# prepruning_finetune_epochs is not used
prepruning_finetune_epochs=1
lagrangian_warmup_epochs=${14}


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    logging_steps=25
    # eval_steps=100
    # epochs=200
    # eval_steps=50
    # epochs=100
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    logging_steps=100
    # QQP 20epoch 227420
    # QNLI 20epoch 65480 eval step 400
    # SST2 20epoch 42100
    # eval_steps=400
    # epochs=40
fi

pretrained_pruned_model=None

# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=${13}
  learning_rate=${12}
  scheduler_type=linear
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=20
  batch_size=64
fi

mkdir -p $output_dir

export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python3 $code_dir/run_glue_prune.py \
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
	   --reg_learning_rate ${reg_learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --target_sparsity $target_sparsity \
     --freeze_embeddings \
     --do_distill \
     --distillation_path $distillation_path \
     --distill_ce_loss_alpha $distill_ce_loss_alpha \
     --distill_loss_alpha $distill_layer_loss_alpha \
     --distill_temp $distill_temp \
     --scheduler_type $scheduler_type \
     --layer_distill_version $layer_distill_version \
     --prepruning_finetune_epochs $prepruning_finetune_epochs \
     --use_mac_l0 \
     --dataloader_num_workers 0 \
     --log_level error \
     --droprate_init 0.01 \
     --prune_location $prune_location \
     --bin_num $bin_num \
     --topk $topk \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
