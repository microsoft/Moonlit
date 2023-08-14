import hashlib
import os
from typing import List

FILE_DIR = os.path.dirname(__file__)

onnx_model_dir = os.path.join(FILE_DIR, '../checkpoints/models/onnx_model')
latency_output_csv_path = os.path.join(FILE_DIR, 'latency.csv')
input_csv_path = os.path.join(FILE_DIR, 'input.csv')


def get_model_name(superspace, supernet_choice, arch):
    if isinstance(supernet_choice, List):
        supernet_choice = ''.join([str(v) for v in supernet_choice])
    arch_encoding = hashlib.sha256(str(arch).encode('utf-8')).hexdigest()[:6]
    rv = f'{superspace}_{supernet_choice}_{arch_encoding}'
    return rv


def get_input_subnets():
    rv = []
    with open(input_csv_path, 'r') as f:
        for line in f.readlines():
            try:
                superspace, supernet_choice, subnet_arch = line.strip().split(',')
                rv.append((superspace, supernet_choice, subnet_arch))
            except:
                pass
    return rv