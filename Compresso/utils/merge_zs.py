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
from utils.compresso_utils import load_zs
from models.modeling_llama import LlamaConfig
from models.model_args import ModelArguments
import torch

logger = logging.getLogger(__name__)


def update_params(lm_model, zs):
    model = lm_model.model

    config = lm_model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers
    # from transformers import BertModel
    # bert = BertModel.from_pretrained("bert-base-uncased")
    if zs is not None:
        # import pdb; pdb.set_trace()
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                if "mlp_z" in zs and zs["mlp_z"][layer] == 0:
                    continue
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                model.layers[layer].mlp.gate_proj.weight.data = model.layers[layer].mlp.gate_proj.weight.transpose(0, 1).data.mul(intermediate_z).transpose(0, 1)
                model.layers[layer].mlp.up_proj.weight.data = model.layers[layer].mlp.up_proj.weight.transpose(0, 1).data.mul(intermediate_z).transpose(0, 1)
                
        if "head_z" in zs:
            for layer in range(num_layers):
                if "head_layer_z" in zs and zs["head_layer_z"][layer] == 0:
                    continue
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                model.layers[layer].self_attn.v_proj.weight.data = model.layers[layer].self_attn.v_proj.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            for layer in range(num_layers):
                model.layers[layer].self_attn.o_proj.weight.data = model.layers[layer].self_attn.o_proj.weight.transpose(0, 1).data.mul(hidden_z).transpose(0, 1)
                model.layers[layer].mlp.down_proj.weight.data = model.layers[layer].mlp.down_proj.weight.transpose(0, 1).data.mul(hidden_z).transpose(0, 1)

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
        #num_labels=num_labels,
        #finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    config.use_cache = False
    config.use_lora = model_args.use_lora

    model = LlamaForCausalLM.from_pretrained(
        LlamaForCausalLM,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
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

    update_params(model, zs)
    
    output_path = "./llama_pruned" if training_args.output_dir == "./" else training_args.output_dir
    model.save_pretrained(output_path)
    print("Save merged Checkpoint! Output path: {output_path}")


if __name__ == "__main__":
    main()
