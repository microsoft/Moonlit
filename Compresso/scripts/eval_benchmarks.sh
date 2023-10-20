#!/bin/bash
export PYTHONPATH='.'

export OUTPUT_DIR=merged_model
mkdir -p $OUTPUT_DIR

base_model=decapoda-research/llama-7b-hf
pretrained_path=output/Compresso-pruning-s30.0-lr5e-05-reglr0.1-warmup4/2023-10-18-10-38/epoch5
prompt_mark=1 # 0: do not add pruning prompt during evaluation; 1: add the pruning prompt same as training; 2. add the pruning prompt for evaluation
lora_param=Q.V # the lora param in training

python ./merge_weights.py \
  --pruning_type None \
  --model_name_or_path $base_model \
  --pretrained_pruned_model $pretrained_path \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param $lora_param \
  --lora_layers 32 \

echo "LoRA and pruning mask MERGED"


cd evaluation/instruct-eval
python main.py mmlu --model_name llama --model_path ../../$OUTPUT_DIR --tokenizer decapoda-research/llama-7b-hf --prompt_mark $prompt_mark
python main.py bbh --model_name llama --model_path ../../$OUTPUT_DIR --tokenizer decapoda-research/llama-7b-hf --prompt_mark $prompt_mark
