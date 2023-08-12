# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# based on AlphaNet

import argparse
import os
import sys

FILE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(FILE_DIR, '..'))
from modules.modeling.supernet import Supernet
from modules.latency_benchmark.tflite.tf2tflite import tf2tflite
from modules.latency_benchmark.tflite.benchmark import tflite_benchmark

from tflite_tools.common import tflite_model_dir, latency_output_csv_path, get_model_name, get_input_subnets


def export():
    for superspace, supernet_choice, subnet_choice in get_input_subnets():
        print(superspace, supernet_choice, subnet_choice)
        model = Supernet.build_from_str('-'.join([superspace, supernet_choice]))

        supernet = model

        supernet.set_active_subnet(subnet_choice)
        subnet = supernet.get_active_tf_subnet()

        model_name = get_model_name(superspace, supernet_choice, subnet_choice)
        fp32_path = os.path.join(tflite_model_dir, f'{model_name}.tflite')
        int8_path = fp32_path.replace('.tflite', '_quant_int8.tflite')
        tf2tflite(subnet, fp32_path)
        tf2tflite(subnet, int8_path, quantization='int8')


def benchmark():
    for superspace, supernet_choice, subnet_choice in get_input_subnets():
        model_name = get_model_name(superspace, supernet_choice, subnet_choice)
        fp32_path = os.path.join(tflite_model_dir, f'{model_name}.tflite')
        int8_path = fp32_path.replace('.tflite', '_quant_int8.tflite')
        fp32_ms, *_ = tflite_benchmark(fp32_path, profiling_output_csv_file=fp32_path.replace('.tflite', '_profiling_output.csv'))
        int8_ms, *_ = tflite_benchmark(int8_path, profiling_output_csv_file=int8_path.replace('.tflite', '_profiling_output.csv'))
        
        speed_up = round(fp32_ms / int8_ms, 2)
        with open(latency_output_csv_path, 'a') as f:
            f.write(f'{superspace}-{supernet_choice}-{subnet_choice},{fp32_ms:.2f},{int8_ms:.2f},{speed_up}\n')


def main():
    export()
    benchmark()


if __name__ == '__main__':
    main()


