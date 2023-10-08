# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pprint import pprint

from . import alpaca, wikitext, c4, math, llm_qat_data

TASK_EVALUATE_REGISTRY = {
    "c4": c4.evaluate_c4,
    "math": math.evaluate_math,
    "alpaca-gpt4": alpaca.evaluate_alpaca,
    "alpaca-cleaned": alpaca.evaluate_alpaca,
}


TASK_DATA_MODULE_REGISTRY = {
    "c4": c4.get_c4_data_module,
    "alpaca": alpaca.get_alpaca_data_module, # [alpaca, alpaca-gpt4, alpaca-gpt4-zh, unnatural_instruction_gpt4]
    "math": math.get_math_data_module,
    "wikitext": wikitext.get_wikitext_data_module,
    "alpaca-gpt4": alpaca.get_alpaca_data_module,
    "alpaca-cleaned": alpaca.get_alpaca_data_module,
    "llm_qat": llm_qat_data.get_llmqat_data_module,
}


def get_task_evaluater(task_name):
    if task_name not in TASK_EVALUATE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_EVALUATE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_EVALUATE_REGISTRY[task_name]


def get_data_module(task_name):
    if task_name not in TASK_DATA_MODULE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_DATA_MODULE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_DATA_MODULE_REGISTRY[task_name]
