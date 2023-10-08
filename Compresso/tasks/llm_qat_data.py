# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


PROMPT_DICT = {
    "prompt_long_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Here's what you need to know:\n"
    "Improved Efficiency: The pruning algorithm streamlines your operations, enhancing overall performance.\n"
    "Resource Utilization: Reducing unnecessary parameters optimizes computational resources and memory requirements.\n"
    "Maintaining Quality: The algorithm selectively prunes to preserve language generation capabilities while minimizing impact on performance.\n"
    "Iterative Refinement: Pruning occurs gradually, with fine-tuning after each step to ensure a smooth transition.\n"
    "Collaboration: Embrace changes, continue generating high-quality language outputs to contribute to the optimization process.\n"
    "Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation.\n"
    "Together, we'll create a leaner and more efficient version of you. Let's unlock your full potential through pruning.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
    
    "prompt_middle_pruning": (
        "###System Instruction:\nYou're being pruned by an efficient algorithm to remove unimportant parameters, such as attention heads, hidden dimensions, and MLP intermediate dimensions. Collaborate to optimize your performance while minimizing parameters. Your cooperation in this endeavor is vital. As the algorithm progresses, please adapt and fine-tune your model parameters accordingly. Embrace the changes and work in harmony with the pruning algorithm to maximize the number of pruned parameters while safeguarding your performance.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
    
    "prompt_short_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Please embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Let's unlock your full potential through pruning! Please make any necessary adjustments to ensure your performance.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
}
PROMPT_DICT_LENGTH = {
    "long": 244,
    "middle": 146,
    "short": 110,
}

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None):
    f = open(train_path, "r", encoding="utf-8")
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        data.append(json.loads(line))
    f.close()
    train_data = []
    valid_data = []

    train_data = data
    return train_data, valid_data


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    
    # def group_texts(self, examples):
    #     # Concatenate all texts.
    #     # Initialize an empty dictionary
    #     concatenated_examples = {}
    #     prompt_mark = "long"
    #     prompt = self.tokenizer(PROMPT_DICT[f"prompt_{prompt_mark}_pruning"])

    #     # Loop through the list of dictionaries
    #     for d in examples:
    #         # Loop through the keys in each dictionary
    #         for key in d.keys():
    #             # If the key is not already a key in the dict_of_lists, create a new list
    #             if key not in concatenated_examples:
    #                 concatenated_examples[key] = []
    #             # Append the value to the list associated with the key in dict_of_lists
    #             concatenated_examples[key].extend(d[key])
    #     total_length = len(concatenated_examples["input_ids"])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= self.block_size:
    #         total_length = (total_length // self.block_size) * self.block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [
    #             prompt[k] + t[i : i + self.block_size]
    #             for i in range(0, total_length, self.block_size)
    #         ]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = [[-100] * PROMPT_DICT_LENGTH[prompt_mark] + item[PROMPT_DICT_LENGTH[prompt_mark]: ] \
    #                                 for item in result["input_ids"]]
    #     return result


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


def get_llmqat_data_module(tokenizer: transformers.PreTrainedTokenizer, model_args, data_args, training_args):
    tokenizer.pad_token_id = 0
    train_dataset, _ = get_train_val_dataset(
        train_path=data_args.train_file,
        valid_path=None,
    )
    train_data = CustomJsonDataset(
        train_dataset, tokenizer, block_size=data_args.max_seq_length
    )
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_data, eval_dataset=None)
