import os

import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm


def build_eval_dataset(dataset_path: str, transform): # dataset_path: imagenet2012 dataset path
    dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)
    return dataset


def to_data_loader(dataset, batch_size, num_workers):
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


def evaluate_onnx(model_path, data_loader, num_threads=4):

    execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # execution_providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, providers=execution_providers, sess_options=session_options)

    input_name = session.get_inputs()[0].name
    model_name = os.path.basename(model_path).split('.')[0]

    total = 0
    total_correct = 0
    with tqdm(total=len(data_loader), desc=f'>> [eval {model_name}]') as t:
        for images, target in data_loader:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            images = images.numpy()
            logits = session.run(None, {input_name:images})[0]
            pred = np.argmax(logits, axis=1)

            num = len(logits)
            correct = np.sum(pred == target.numpy())
            total += num
            total_correct += correct

            t.set_postfix(total_acc=f'{total_correct/total*100: .1f}%', local_acc=f'{correct/num*100: .1f}%')
            t.update()

    accuracy = total_correct / total * 100
    print(f'Evaluate accuracy {model_name} {accuracy:.1f}%')
    return accuracy


def build_eval_transform(resize_size=256, image_size=224, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def evaluate_onnx_pipeline(model_path, dataset_path, transform=build_eval_transform(), num_threads=8, batch_size=100, num_workers=4):
    dataset = build_eval_dataset(dataset_path, transform)
    data_loader = to_data_loader(dataset, batch_size, num_workers)
    return evaluate_onnx(model_path, data_loader, num_threads)