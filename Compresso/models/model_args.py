# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='/dev/shm',
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    training_objective: str = field(
        default="CLS",
        metadata={"help": "The training objective of bloom to use (CLS or CLM) for finetuning downstream model."}
    )
    template: str = field(
        default="Does the query \"{}\" match the keyword \"{}\"? Answer:",
        metadata={"help": "The template (prompt) to tune downstream QK task using CLM objective."}
    )
    use_features: bool = field(
        default=True,
        metadata={"help": "Whether to use data features or not."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA or not."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The rank of LoRA."},
    )
    lora_train_bias: str = field(
        default="none",
        metadata={"help": "Whether to train bias, choices: none, lora_only and all."},
    )
    lora_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha of LoRA when adding the LoRA parameters to the origin ones."},
    )
    lora_param: str = field(
        default="Q.V",
        metadata={
            "help": "The parameter groups to apply LoRA, including E (embeddings), Q (attn query), K (attn key), "
                    "V (attn value), O (attn output) and F (feedforward), splitted by dot, e.g. Q.V means applying "
                    "LoRA to Q and V."
        }
    )
    lora_layers: int = field(
        default=-1,
        metadata={"help": "The number of top layers to apply LoRA. Set to -1 to apply LoRA to all layers."},
    )
    random_init: bool = field(
        default=False,
        metadata={"help": "If true, do not load the pretained weights and initialize the model with random weights."},
    )

