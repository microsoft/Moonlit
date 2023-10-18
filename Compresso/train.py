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
from trainer.compresso_trainer import CompressoTrainer
from models.l0_module import L0Module
from args import AdditionalArguments, DataTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from models.modeling_llama import LlamaForCausalLM
from utils.compresso_utils import load_zs, initialize_layer_transformation
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer
from models.model_args import ModelArguments
import torch
import deepspeed

logger = logging.getLogger(__name__)

ALPACA_TASK = ["alpaca", "alpaca-gpt4", "alpaca-gpt4-zh", "unnatural_instruction_gpt4", "math", "open_orca", "alpaca-cleaned"]

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config


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
    additional_args.eval_dataset_name = additional_args.eval_dataset_name.split(",")

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
        )
        config.use_cache = False
        lora_ckpt = None
        config = set_lora_args(config, model_args)
        if additional_args.pretrained_pruned_model is not None:
            lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')
            logger.info(f"Load lora ckpt from {lora_ckpt}")
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
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
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                lora_ckpt = lora_ckpt
            )
    else:
        raise ValueError("Training objective should be either cls or clm")
    
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    l0_module = None
    if additional_args.pruning_type is not None:
        l0_module = L0Module(config=config,
                            droprate_init=additional_args.droprate_init,
                            layer_gate_init_open=additional_args.layer_gate_init_open,
                            layer_gate_open_0=additional_args.layer_gate_open_0,
                            block_layer_start=additional_args.block_layer_start,
                            block_layer_end=additional_args.block_layer_end,
                            sparsity_scheduler=additional_args.sparsity_scheduler,
                            temperature=additional_args.temperature,
                            target_sparsity=additional_args.target_sparsity,
                            pruning_type=additional_args.pruning_type)

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
        logger.info(f"Load pretrained zs!")
        for key in zs:
            zs[key] = zs[key].detach()
        
        if zs["head_z"].shape[0] < config.num_hidden_layers:
            if zs["head_z"].shape[0] == 26:
                zs["head_z"] = torch.concat([torch.ones(4, 1, 32, 1, 1), zs["head_z"], torch.ones(2, 1, 32, 1, 1)])
                zs["intermediate_z"] = torch.concat([torch.ones(4, 1, 1, 11008), zs["intermediate_z"], torch.ones(2, 1, 1, 11008)])
            elif zs["head_z"].shape[0] == 28:
                zs["head_z"] = torch.concat([torch.ones(3, 1, 32, 1, 1), zs["head_z"], torch.ones(1, 1, 32, 1, 1)])
                zs["intermediate_z"] = torch.concat([torch.ones(3, 1, 1, 11008), zs["intermediate_z"], torch.ones(1, 1, 1, 11008)])

        if "layer_z" in zs:
            zs['head_layer_z'] = zs['layer_z']
            zs['mlp_z'] = zs['layer_z']
            zs.pop('layer_z')

    # dataset initialize
    from tasks import get_data_module
    if data_args.dataset_name in ALPACA_TASK:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args, model)
    else:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args)
    # use wikitext2 test dataset to evaluate the performance of model on alpaca or math10k
    wiki_module = get_data_module(additional_args.eval_dataset_name[0] if "wikitext" in additional_args.eval_dataset_name[0] else "wikitext")(tokenizer, model_args, data_args, training_args)
    data_module['eval_dataset'] = wiki_module['eval_dataset']
    data_module['compute_metrics'] = wiki_module['compute_metrics']
    data_module['preprocess_logits_for_metrics'] = wiki_module['preprocess_logits_for_metrics']
    # Initialize our Trainer
    trainer = CompressoTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        tokenizer=tokenizer,
        use_lora=model_args.use_lora,
        lora_train_bias=model_args.lora_train_bias,
        l0_module=l0_module,
        **data_module
    )

    from transformers.integrations import AzureMLCallback, ProgressCallback
    trainer.remove_callback(AzureMLCallback)
    trainer.remove_callback(ProgressCallback)
    logger.info("Remove AzureMLCallback and ProgressCallback in Trainer.")

    if additional_args.pretrained_pruned_model is not None:
        trainer.zs = zs

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(None)

    # Evaluating
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=data_module["eval_dataset"])
        trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    '''
    example command:
    
    # wikitext-2
    python train.py --output_dir ./.tmp --model_name_or_path /home/jiahangxu/working/llama/7B_converted --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --max_seq_length 1024 --do_train --training_objective LM
    '''
    os.environ["WANDB_DISABLED"] = "true"
    main()
