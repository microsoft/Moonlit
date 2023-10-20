#!/bin/bash
export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

baseline_pruned_model=output/Compresso-pruning-s30.0-lr5e-05-reglr0.1-warmup4/2023-10-18-10-38/epoch5

# run finetune
# use ``pretrained_pruned_model`` to clarify zs.pt
deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type None \
  --target_sparsity 0.3 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model $baseline_pruned_model \
  --num_train_epochs 3 \
  --learning_rate 4e-4 \
  --reg_learning_rate 0.05 \
  --lagrangian_warmup_epochs 0 \
  --max_seq_length 1024 \
  --task_name finetune \
  --do_train \
  --do_eval \
  --dataset_name alpaca-gpt4 \
  --eval_dataset_name wikitext \
  --train_file ./data/alpaca_gpt4_data.json \
  --droprate_init 0.01 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 8 \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False \
  --lr_scheduler_type cosine
