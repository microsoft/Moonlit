import argparse
import os
import pathlib
import sys 

import numpy as np
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantFormat, QuantType
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

FILE_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, os.path.join(FILE_DIR, '../../..'))
from modules.latency_benchmark.onnx.quant_onnx import ImageNetCalibrationDataReader
from modules.latency_benchmark.onnx.export_onnx import export_onnx
from modules.latency_benchmark.onnx.eval import evaluate_onnx_pipeline, build_eval_transform


np.random.seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_path', default='path_to_imagenet', type=str, help='imagenet2012 dataset path')
    parser.add_argument('--resolution', type=int)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    evaluate_onnx_pipeline(args.model_path, args.dataset_path, transform=transform)


if __name__ == '__main__':
    main()
    