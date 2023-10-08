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

ALPACA_TASK = ["alpaca", "alpaca-gpt4", "alpaca-gpt4-zh", "unnatural_instruction_gpt4", "math", "open_orca", "alpaca-cleaned"]

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config

# refered_files_path = "~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4/llama_pruned"

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
            use_auth_token="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",
        )
        config.use_cache = False
        lora_ckpt = None
        config = set_lora_args(config, model_args)
        # When runing Finetune, use lora merged "llama_pruned" as base model and do NOT load lora_ckpt
        if additional_args.pretrained_pruned_model is not None and "llama_pruned" not in model_args.model_name_or_path:
            lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')
            logger.info(f"load lora ckpt from {lora_ckpt}")
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
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model,'zs.pt'))
        # import pdb; pdb.set_trace()
        for key in zs:
            zs[key] = zs[key].detach()
        # l0_module = torch.load(os.path.join(additional_args.pretrained_pruned_model,'l0_module.pt'), map_location="cpu")
        # zs = l0_module.forward(training=False)
        # l0_module = None
        
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
        #zs.pop('gate_layer_z')
        #model = load_model(additional_args.pretrained_pruned_model, OPTForCausalLM, zs)
        print(
            f"Model Size after pruning: {calculate_parameters(model)}")

    # zs.pop("head_z")
    
    update_params(model, zs)
    
    config.use_lora = False
    llama = LlamaForCausalLM.from_pretrained(LlamaForCausalLM, model_args.model_name_or_path, config=config)
    
    # import pdb; pdb.set_trace()
    # # input = model.model.embed_tokens(torch.tensor([[1,2,3,8]]))
    # # print(model.model.layers[0].mlp(input, None, None))
    # # print(llama.model.layers[0].mlp(input,intermediate_z=zs["intermediate_z"][0],mlp_z=zs["mlp_z"][0]))
    
    # print(model(torch.tensor([[1,2,3,8]])).logits)
    # print(llama(torch.tensor([[1,2,3,8]]), **zs).logits)
    
    # inputs = {"position_ids": torch.tensor([[1,2,3,4]]), "attention_mask":torch.ones((1,1,4,4))}
    # print(model.model.layers[0].forward(input, **inputs))
    
    output_path = "./llama_pruned"
    llama.load_state_dict(model.state_dict(), strict=False)
    llama.save_pretrained(output_path)
    
    # os.system(f"cp {refered_files_path}/tokenizer.model {output_path}")
    # os.system(f"cp {refered_files_path}/tokenizer.json {output_path}")
    # os.system(f"cp {refered_files_path}/tokenizer_config.json {output_path}")
    # os.system(f"cp {refered_files_path}/special_tokens_map.json {output_path}")
    
    
    input = torch.tensor([[1,2,3,8]])
    a1 = model.model(input).logits
    b1 = llama.model(input, **zs).logits
    print(model(input, **zs))
    print(llama(input))
    
    
    # llama.model.embed_tokens(input, **zs)


    # # dataset initialize
    # from tasks import get_data_module
    # if data_args.dataset_name in ALPACA_TASK:
    #     data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args, model)
    # else:
    #     data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args)
    # # use wikitext2 test dataset to evaluate the performance of model on alpaca or math10k
    # wiki_module = get_data_module(additional_args.eval_dataset_name[0] if "wikitext" in additional_args.eval_dataset_name[0] else "wikitext")(tokenizer, model_args, data_args, training_args)
    # data_module['eval_dataset'] = wiki_module['eval_dataset']
    # data_module['compute_metrics'] = wiki_module['compute_metrics']
    # data_module['preprocess_logits_for_metrics'] = wiki_module['preprocess_logits_for_metrics']
    # # Initialize our Trainer
    # trainer = CoFiTrainer(
    #     model=model,
    #     args=training_args,
    #     additional_args=additional_args,
    #     tokenizer=tokenizer,
    #     use_lora=model_args.use_lora,
    #     lora_train_bias=model_args.lora_train_bias,
    #     l0_module=l0_module,
    #     **data_module
    # )

    # from transformers.integrations import AzureMLCallback, ProgressCallback
    # trainer.remove_callback(AzureMLCallback)
    # trainer.remove_callback(ProgressCallback)
    # logger.info("Remove AzureMLCallback and ProgressCallback in Trainer.")

    # if additional_args.pretrained_pruned_model is not None:
    #     trainer.zs = zs

    # # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(None)

    # # Evaluating
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(eval_dataset=data_module["eval_dataset"])
    #     trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    '''
    example command:
    
    # wikitext-2
    python train.py --output_dir ./.tmp --model_name_or_path /home/jiahangxu/working/llama/7B_converted --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --max_seq_length 1024 --do_train --training_objective LM
    '''
    os.environ["WANDB_DISABLED"] = "true"
    main()
