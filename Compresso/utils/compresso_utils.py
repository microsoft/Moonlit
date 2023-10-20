# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import os
from transformers import AutoConfig

from utils.utils import calculate_parameters


def edit_config(config, additional_args):
    config.transform_embedding = additional_args.transform_embedding
    config.do_distill = additional_args.do_distill
    config.do_layer_distill = additional_args.do_layer_distill


def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)


def load_model_with_zs(model_path, model_class, zs=None):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=config)
    p = os.path.join(model_path, "pytorch_model.bin")
    loaded_weights = torch.load(p, map_location="cpu")
    model.load_state_dict(loaded_weights)
    print(f"Load weights from {model_path}")
    return model


def load_model(model_path, model_class, zs=None):
    assert zs is not None
    model = load_model_with_zs(model_path, model_class, zs)
    return model


# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None


def load_zs(model_path):
    if model_path.endswith("zs.pt"):
        zs_path = model_path
    else:
        zs_path = os.path.join(model_path, "zs.pt")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(os.path.join(model_path, "l0_module.pt"), map_location="cpu")
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None

def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name)
    model_size = calculate_parameters(model)
    return model_size
