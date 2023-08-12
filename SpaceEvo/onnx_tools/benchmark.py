# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# based on AlphaNet

import argparse
import os
import sys
import random
from datetime import date, datetime

import numpy as np
import torch

FILE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(FILE_DIR, '..'))
from modules.latency_benchmark.onnx.benchmark import onnx_benchmark

from onnx_tools.common import onnx_model_dir, latency_output_csv_path, get_model_name, get_input_subnets


parser = argparse.ArgumentParser()
args = parser.parse_args()


def main():

    for superspace, supernet_choice, subnet_choice in get_input_subnets():
        model_name = get_model_name(superspace, supernet_choice, subnet_choice)
        resolution = int(subnet_choice[-3:])
        fp32_path = os.path.join(onnx_model_dir, f'{model_name}.onnx')
        int8_path = fp32_path.replace('.onnx', '_quant_int8.onnx')
        fp32_ms, _ = onnx_benchmark(fp32_path, num_runs=100, warmup_runs=100)
        int8_ms, _ = onnx_benchmark(int8_path, num_runs=100, warmup_runs=100)
        speed_up = round(fp32_ms / int8_ms, 2)
        with open(latency_output_csv_path, 'a') as f:
            f.write(f'{superspace}-{supernet_choice}-{subnet_choice},{fp32_ms},{int8_ms},{speed_up}\n')

if __name__ == '__main__':
    main()


