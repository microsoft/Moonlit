# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os
import sys
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.cofi_utils import initialize_layer_transformation
from trainer.cofi_trainer import CoFiTrainer
from models.l0_module import L0Module
from args import AdditionalArguments, DataTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from models.modeling_llama import LlamaForCausalLM
from utils.utils import calculate_parameters
from utils.cofi_utils import load_zs
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer
from models.model_args import ModelArguments
import torch
import deepspeed

logger = logging.getLogger(__name__)

ALPACA_TASK = ["alpaca", "alpaca-gpt4", "alpaca-gpt4-zh", "unnatural_instruction_gpt4", "math", "open_orca"]

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config

# refered_files_path = "/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4/llama_pruned"

def main():
    # # Used for profiling, usage:
    # #   [install] sudo env "PATH=$PATH" pip install viztracer
    # #   [profile] sudo env "PATH=$PATH" viztracer --attach_installed [PID]
    # from viztracer import VizTracer
    # tracer = VizTracer()
    # tracer.install()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    training_args.report_to = []

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args} \n {additional_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # model initialize
    if model_args.training_objective == "LM":
        config = LlamaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            #num_labels=num_labels,
            #finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token="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",
        )
        config.use_cache = False
        lora_ckpt = None
        config = set_lora_args(config, model_args)
        if additional_args.pretrained_pruned_model is not None:
            lora_ckpt = os.path.join(additional_args.pretrained_pruned_model,'lora_weights.pt')
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token="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",
            padding_side="left",
            truncation_side="left",
        )
        if model_args.random_init:
            from transformers.deepspeed import deepspeed_config
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = LlamaForCausalLM(
                    config=config,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                LlamaForCausalLM,
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token="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",
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                lora_ckpt = lora_ckpt
            )
    else:
        raise ValueError("Training objective should be either cls or clm")
    
    config.use_lora = False
    llama = LlamaForCausalLM.from_pretrained(LlamaForCausalLM, model_args.model_name_or_path, config=config)
    output_path = "./llama_pruned"
    llama.load_state_dict(model.state_dict(), strict=False)
    llama.save_pretrained(output_path)
    os.system(f"cp {refered_files_path}/tokenizer.model {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer.json {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer_config.json {output_path}")
    os.system(f"cp {refered_files_path}/special_tokens_map.json {output_path}")
    print("Save merged Checkpoint!")
    
    
if __name__ == "__main__":
    '''
    example command:
    
    # wikitext-2
    python train.py --output_dir ./.tmp --model_name_or_path /home/jiahangxu/working/llama/7B_converted --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --max_seq_length 1024 --do_train --training_objective LM
    '''
    os.environ["WANDB_DISABLED"] = "true"
    main()
