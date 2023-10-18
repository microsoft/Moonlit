# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os
import sys
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from args import AdditionalArguments
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama import LlamaConfig
from models.model_args import ModelArguments

logger = logging.getLogger(__name__)


def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )    
    training_args.report_to = []

    # model initialize
    config = LlamaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    config.use_cache = False
    config = set_lora_args(config, model_args)
    lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')

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
    
    config.use_lora = False
    llama = LlamaForCausalLM.from_pretrained(LlamaForCausalLM, model_args.model_name_or_path, config=config)
    output_path = "./llama_pruned" if training_args.output_dir == "./" else training_args.output_dir
    llama.load_state_dict(model.state_dict(), strict=False)
    llama.save_pretrained(output_path)
    print("Save merged Checkpoint! Output path: {output_path}")


if __name__ == "__main__":
    main()
