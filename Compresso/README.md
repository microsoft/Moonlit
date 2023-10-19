<h1 align="center"> 
<p> Compresso </p>
</h1>

## Install dependencies
```bash
pip install -r requirements.txt
```

## Pruning

**Step 1**: Prepare dataset for training: Download [Alpaca GPT4 dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) and put the dataset into ``./data``.

**Step 2**: Run ``./scripts/train.sh`` to try Compresso Pruning.

```bash
deepspeed --num_nodes=1 --num_gpus=$NUM_GPUS train.py \
    --deepspeed ds3_offload.json \
    --pruning_type structured_heads+structured_mlp+hidden \
    --target_sparsity 0.3 \
    --sparsity_epsilon 0.005 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --num_train_epochs 8 \
    --learning_rate 5e-5 \
    --reg_learning_rate 0.1 \
    --lagrangian_warmup_epochs 4 \
    --max_seq_length 1024 \
    --task_name pruning \
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
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm True \
    --fp16 false \
    --random_init=False
```

## Finetuning

**Step 1**: After pruning, prepare the pruned output folder as ``$baseline_pruned_model``, it should consist of LoRA weights file ``lora_weights.pt`` and pruning mask file ``zs.pt`.

**Step 2**: Run ``./scripts/finetune.sh`` to try Compresso Pruning.
``` bash
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
```

## Evaluation

We evaluate Compresso on the LLaMA family. We follow the original LLaMA paper to measure the effectiveness of pruned LLMs across three key application domains:

- [Zero-shot Commonsense Reasoning](./lm-evaluation-harness/). We evaluate the 0-shot results for 7 commonsense reasoning benchmarks: StoryCloze, PIQA, HellaSwag, WinoGrande, ARC easy and challenge, and OpenBookQA (OBQA). [[Example link](../scripts/eval_commonsense.sh)]
    > Note: StoryCLoze dataset requires manual download of data. Users should clarify the data folder path in line 418 of [code](./lm-evaluation-harness/lm_eval/tasks/__init__.py)

- [Reading Comprehension](./lm-evaluation-harness/). We also evaluate the 0-shot performance on two reading comprehension benchmarks: BoolQ and RACE-High. [[Example link](../scripts/eval_reading.sh)]

- [Popular Aggregated Benchmarks](./instruct-eval/). Besides, we evaluate the in-context learning ability under a few-shot setting. We report the results on MMLU (5 shot), which consists of 57 tasks covering STEM, humanities, social science, etc, and Big Bench Hard (BBH) (3 shot), which includes 23 challenging tasks. [[Example link](../scripts/eval_benchmarks.sh)]
    > Note: We used the Compresso model with LoRA weights and pruning mask merged to perform this evaluation. Since the merged Compresso structure (without truly pruned) is identical to the original Llama, we can directly use the pipeline same as LLaMA for evaluation. The merging step could be completed using ``./utils/merge_weights.py`` and ``./utils/merge_zs.py``. For more details, please refer to the [example](../scripts/eval_reading.sh).
    > Note: instruct-eval repository requires to create new virtual environment, and within the new environment, ``transformers==4.25.1`` will to be updated to ``transformers==4.29.0``.

For commonsense reasoning and reading comprehension, we use [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master)
to carry out the evaluations. For MMLU and BBH, we use [InstructEval](https://github.com/declare-lab/instruct-eval/tree/main). We saved the repo versions used in our work within this folder in case any changes in the results due to future updates in these repos.

- lm-evaluation-harness setup:

``` bash
cd evaluation/lm-evaluation-harness
pip install -e .
```

- InstructEval setup:

``` bash
conda create -n instruct-eval python=3.8 -y
conda activate instruct-eval
pip install -r requirements.txt
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```
