export PYTHONPATH='.'

base_model=decapoda-research/llama-7b-hf
pretrained_path=output/Compresso-pruning-s30.0-lr5e-05-reglr0.1-warmup4/2023-10-18-10-38/epoch5
prompt_mark=1 # 0: do not add pruning prompt during evaluation; 1: add the pruning prompt same as training; 2. add the pruning prompt for evaluation
lora_param=Q.V # the lora param in training

file_name=$(echo $pretrained_path | cut -d'/' -f $(($(echo $pretrained_path | tr '/' '\n' | wc -l) - 2)))
python ./evaluation/lm-evaluation-harness/main.py \
    --model compresso \
    --model_args pretrained=$base_model,peft=$pretrained_path,prompt_mark=$prompt_mark,lora_param=$lora_param \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/Zeroshot_${file_name}_$prompt_mark.json \
    --no_cache

python ./evaluation/lm-evaluation-harness/generate.py results/${file_name}_$prompt_mark.json
