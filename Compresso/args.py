# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass, field
from typing import Optional

from tasks import TASK_DATA_MODULE_REGISTRY

@dataclass
class AdditionalArguments():
    test: bool = field(
        default=False,
        metadata={
            "help": "Testing additional arguments."
        },
    )

    ex_name: str = field(default="test", metadata={"help": "Name of experiment. Base directory of output dir."})
    pruning_type: str = field(default=None, metadata={"help": "Type of pruning"})
    reg_learning_rate: float = field(default=0.1, metadata={"help": "Learning rate for regularization."})
    scheduler_type: str = field(default="linear", metadata={"help": "type of scheduler"})
    freeze_embeddings: bool = field(default=False, metadata={"help": "Whether we should freeze the embeddings."})

    pretrained_pruned_model: str = field(default=None, metadata={"help": "Path of pretrained model."})

    droprate_init: float = field(default=0.5, metadata={"help": "Init parameter for loga"})
    layer_gate_init_open: bool = field(default=False, metadata={"help": "Whether to open all layer gate when init."})
    layer_gate_open_0: bool = field(default=False, metadata={"help": "Layer gate open 0: pruned model sparsity smaller; Layer gate open 1: pruned model sparsity larger;"})
    temperature: float = field(default=2./3., metadata={"help": "Temperature controlling hard concrete distribution"})
    prepruning_finetune_epochs: int = field(default=1, metadata={"help": "Finetuning epochs before pruning"})
    lagrangian_warmup_epochs: int = field(default=2, metadata={"help": "Number of epochs for lagrangian warmup"})
    target_sparsity: float = field(default=0, metadata={"help": "Target sparsity (pruned percentage)"})
    sparsity_epsilon: float = field(default=0, metadata={"help": "Epsilon for sparsity"})
    block_layer_start: int = field(default=None)
    block_layer_end: int = field(default=None)
    sparsity_scheduler: str = field(default="linear")

    task_name: Optional[str] = field(default=None,metadata={"help": "The name of the task to train on: " + ", ".join(TASK_DATA_MODULE_REGISTRY.keys())})
    # distillation setup
    distillation_path: str = field(default=None, metadata={"help": "Path of the teacher model for distillation."})
    do_distill: bool = field(default=False, metadata={"help": "Whether to do distillation or not, prediction layer."})
    do_layer_distill: bool = field(default=False, metadata={"help": "Align layer output through distillation"})
    layer_distill_version: int = field(default=1, metadata={"help": "1: add loss to each layer, 2: add loss to existing layers only"})
    distill_loss_alpha: float = field(default=0.9, metadata={"help": "Distillation loss weight"})
    distill_ce_loss_alpha: float = field(default=0.1, metadata={"help": "Distillation cross entrypy loss weight"})
    distill_temp: float = field(default=2./3., metadata={"help": "Distillation temperature"})

    # evaluation arguments for math-related dataset
    max_eval_math_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use."}
    )
    eval_prompt_type: Optional[int] = field(
        default=0, metadata={"help": "Whether add prompt in zero-shot evaluation, 0: no prompt; 1: prompt long; "}
    )
    eval_method: Optional[str] = field(
        default="few_shot_cot",
        # choices=["zero_shot", "few_shot", "few_shot_cot"],
        metadata={"help": "COT method."}
    )
    cot_trigger_no: Optional[int] = field(
        default=1, metadata={"help": "A trigger sentence that elicits a model to execute chain of thought."}
    )
    cot_shot_length: Optional[int] = field(
        default=8, metadata={"help": "length of shots of cot for few-shot ICL settings."}
    )
    max_length_cot: Optional[int] = field(
        default=512, metadata={"help": "maximum length of output tokens by model for reasoning extraction."}
    )
    max_length_direct: Optional[int] = field(
        default=32, metadata={"help": "maximum length of output tokens by model for answer extraction."}
    )

    def __post_init__(self):
        if self.pretrained_pruned_model == "None":
            self.pretrained_pruned_model = None
        if self.pruning_type == "None":
            self.pruning_type = None


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    prompt_mark: Optional[str] = field(default="long", metadata={"help": "choose from ['long', 'short', 'middle']. (If additional_args.pruning_prompt_stage=0, this mark will not be used)"})
