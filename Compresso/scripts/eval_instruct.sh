#!/bin/bash
export PYTHONPATH='.'

export OUTPUT_DIR=merged_model
mkdir -p $OUTPUT_DIR

base_model=decapoda-research/llama-7b-hf
pretrained_path=output/Compresso-pruning-s30.0-lr5e-05-reglr0.1-warmup4/2023-10-18-10-38/epoch5
prompt_mark=1 # 0: do not add pruning prompt during evaluation; 1: add the pruning prompt same as training; 2. add the pruning prompt for evaluation
lora_param=Q.V # the lora param in training

deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 utils/merge_weights.py \
  --pruning_type None \
  --model_name_or_path $base_model \
  --pretrained_pruned_model $pretrained_path \
  --task_name None \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param $lora_param \
  --lora_layers 32 \

echo "STEP 1 FINISHED"
# python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./llama_pruned --prompt_mark 0


python utils/merge_zs.py \
  --pruning_type None \
   --model_name_or_path $OUTPUT_DIR \
  --pretrained_pruned_model $pretrained_path \
  --task_name None \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
  --use_lora False \

echo "STEP 2 FINISHED"
# python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./llama_pruned --prompt_mark 0


cd evaluation/instruct-eval
python main.py mmlu --model_name llama --model_path ../../$OUTPUT_DIR --prompt_mark $prompt_mark
python main.py bbh --model_name llama --model_path ../../$OUTPUT_DIR --prompt_mark $prompt_mark
