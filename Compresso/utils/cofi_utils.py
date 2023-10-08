# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import os
from transformers.modeling_utils import prune_linear_layer
from transformers import AutoConfig, BertForSequenceClassification
# from transformers.file_utils import hf_bucket_url, cached_path

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

    update_params(model, zs)
    print(f"Model Size before pruning: {calculate_parameters(model)}")
    prune_model_with_z(zs, model)
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model

def load_model(model_path, model_class, zs=None):
    assert zs is not None
    model = load_model_with_zs(model_path, model_class, zs)
    print(f"Model Size: {calculate_parameters(model)}")
    return model

# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None

# z values could be in [0, 1), we update the parameters accordingly with z values
def update_params(model, zs):
    #bert = model.bert if hasattr(model, "bert") else model.roberta
    model = model.model
    config = model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers

    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                model.decoder.layers[layer].fc2.weight.data = model.decoder.layers[layer].fc2.weight.data.mul(intermediate_z)
                #bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)
                if "mlp_z" in zs:
                    mlp_z = zs["mlp_z"][layer].cpu()
                    model.decoder.layers[layer].fc2.weight.data = model.decoder.layers[layer].fc2.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
                    model.decoder.layers[layer].fc2.bias.data = model.decoder.layers[layer].fc2.bias.data.mul(mlp_z)
                    #bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
                    #bert.encoder.layer[layer].output.dense.bias.data = bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)

        if "head_z" in zs:
            for layer in range(num_layers):
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                model.decoder.layers[layer].self_attn.v_proj.weight.data = model.decoder.layers[layer].self_attn.v_proj.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)
                model.decoder.layers[layer].self_attn.v_proj.bias.data = model.decoder.layers[layer].self_attn.v_proj.bias.data.mul(head_z)
                #bert.encoder.layer[layer].attention.self.value.weight.data = bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)
                #bert.encoder.layer[layer].attention.self.value.bias.data = bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)
                if "head_layer_z" in zs:
                    head_layer_z = zs["head_layer_z"][layer].cpu()
                    model.decoder.layers[layer].self_attn.out_proj.weight.data = model.decoder.layers[
                        layer].self_attn.out_proj.weight.transpose(0, 1).data.mul(head_layer_z).transpose(0, 1)
                    model.decoder.layers[layer].self_attn.out_proj.bias.data = model.decoder.layers[
                        layer].self_attn.out_proj.bias.data.mul(head_layer_z)
                    # bert.encoder.layer[layer].attention.output.dense.weight.data = bert.encoder.layer[
                    #     layer].attention.output.dense.weight.transpose(0, 1).data.mul(head_layer_z).transpose(0, 1)
                    # bert.encoder.layer[layer].attention.output.dense.bias.data = bert.encoder.layer[
                    #     layer].attention.output.dense.bias.data.mul(head_layer_z)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            model.decoder.embed_positions.weight.data = model.decoder.embed_positions.weight.data.mul(hidden_z)
            model.decoder.embed_tokens.weight.data = model.decoder.embed_tokens.weight.data.mul(hidden_z)
            #     bert.embeddings.word_embeddings.weight.data.mul(hidden_z)
            # bert.embeddings.position_embeddings.weight.data = \
            #     bert.embeddings.position_embeddings.weight.data.mul(hidden_z)
            # bert.embeddings.token_type_embeddings.weight.data = \
            #     bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)
            for layer in range(num_layers):
                model.decoder.layers[layer].self_attn.k_proj.weight.data = model.decoder.layers[layer].self_attn.k_proj.weight.data.mul(hidden_z)
                model.decoder.layers[layer].self_attn.q_proj.weight.data = model.decoder.layers[layer].self_attn.q_proj.weight.data.mul(hidden_z)
                model.decoder.layers[layer].self_attn.v_proj.weight.data = model.decoder.layers[layer].self_attn.v_proj.weight.data.mul(hidden_z)
                model.decoder.layers[layer].self_attn.out_proj.weight.data = model.decoder.layers[layer].self_attn.out_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
                model.decoder.layers[layer].self_attn.out_proj.bias.data = model.decoder.layers[layer].self_attn.out_proj.bias.data.mul(hidden_z)
                model.decoder.layers[layer].fc1.weight.data = model.decoder.layers[layer].fc1.weight.data.mul(hidden_z)
                model.decoder.layers[layer].fc2.weight.data = model.decoder.layers[layer].fc2.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            # if hasattr(model.pooler, "dense"):
            #     bert.pooler.dense.weight.data = bert.pooler.dense.weight.data.mul(hidden_z)
            if hasattr(model, "qa_outputs"):
                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)


def prune_model_with_z(zs, model):
    if zs is None:
        return None, None
    #bert = model.bert if hasattr(model, "bert") else model.roberta
    model_p = model
    model = model_p.model
    if "head_z" in zs:
        head_z = zs.get("head_z", None)
        head_layer_z = zs.get("head_layer_z", None)

        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer *= head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index

            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(prune_heads)


    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z", None)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer *= mlp_z[layer]
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()

    def prune_layer_norm(layernorm, index):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index))
        layernorm.bias = torch.nn.parameter.Parameter(
            layernorm.bias.index_select(0, index))
        layernorm.normalized_shape = (len(index),)

    def prune_layer(layer, index, dim):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer

    if "hidden_z" in zs:
        hidden_zs = zs["hidden_z"]
        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        # bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
        #     bert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach())
        # bert.embeddings.word_embeddings.embedding_dim = index.shape[0]
        model.decoder.embed_positions.weight = torch.nn.parameter.Parameter(
            model.decoder.embed_positions.weight.index_select(1, index).clone().detach())
        model.decoder.embed_positions.embedding_dim = index.shape[0]
        model.decoder.embed_tokens.weight = torch.nn.parameter.Parameter(
            model.decoder.embed_tokens.weight.index_select(1, index).clone().detach())
        model.decoder.embed_tokens.embedding_dim = index.shape[0]
        prune_layer_norm(model.decoder.final_layer_norm, index)

        for layer in range(0, len(model.decoder.layers)):
            if model.decoder.layers[layer].self_attn.q_proj is not None:
                model.decoder.layers[layer].self_attn.q_proj = \
                    prune_layer(model.decoder.layers[layer].self_attn.q_proj , index, dim=1)
                model.decoder.layers[layer].self_attn.k_proj = \
                    prune_layer(model.decoder.layers[layer].self_attn.k_proj , index, dim=1)
            if model.decoder.layers[layer].self_attn.v_proj is not None:
                model.decoder.layers[layer].self_attn.v_proj = \
                    prune_layer(model.decoder.layers[layer].self_attn.v_proj , index, dim=1)
                model.decoder.layers[layer].self_attn.out_proj = \
                    prune_layer(model.decoder.layers[layer].self_attn.out_proj , index, dim=0)
            prune_layer_norm(model.decoder.layers[layer].self_attn_layer_norm, index)
            if model.decoder.layers[layer].fc1 is not None:
                model.decoder.layers[layer].fc1 = \
                    prune_layer( model.decoder.layers[layer].fc1, index, dim=1)
                model.decoder.layers[layer].fc2 = \
                    prune_layer( model.decoder.layers[layer].fc2, index, dim=0)
            prune_layer_norm(model.decoder.layers[layer].final_layer_norm, index)
            

        # accommodate for different models
        # if hasattr(model, "classifier"):
        #     if hasattr(model.classifier, "dense"):
        #         model.classifier.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        # if hasattr(model, "cls"):
        #     if hasattr(model.cls, "dense"):
        #         model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        # if hasattr(bert.pooler, "dense"):
        #     bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)
        # if hasattr(model, "qa_outputs"):
        #     model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(model.layer_transformation, index, dim=1)
            print("layer transformation", model.layer_transformation.weight.shape)
        if getattr(model, "mha_layer_transformation", None) is not None:
            model.mha_layer_transformation = prune_linear_layer(model.mha_layer_transformation, index, dim=1)
            print("layer mha_layer_transformation", model.mha_layer_transformation.weight.shape)
        if hasattr(model_p, "lm_head"):
            model_p.lm_head = prune_linear_layer(model_p.lm_head, index, dim=1)
    if kept_intermediate_dims is not None:
        prune_intermediate_layers(model, kept_intermediate_dims)

    for layer in range(0, len(model.decoder.layers)):
        print("Layer:", layer)
        if model.decoder.layers[layer].self_attn.q_proj is not None:
            print("query:", model.decoder.layers[layer].self_attn.q_proj.weight.shape)
            print("key:", model.decoder.layers[layer].self_attn.k_proj.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if model.decoder.layers[layer].self_attn.v_proj is not None:
            print("value:", model.decoder.layers[layer].self_attn.v_proj.weight.shape)
            print("output:", model.decoder.layers[layer].self_attn.out_proj.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if model.decoder.layers[layer].fc1 is not None:
            print("up:", model.decoder.layers[layer].fc1.weight.shape)
            print("down:", model.decoder.layers[layer].fc2.weight.shape)
        else:
            print("up", None)
            print("down", None)


def prune_intermediate_layers(model, keep_dims):
    #bert = model.bert if hasattr(model, "bert") else model.roberta
    device = model.device
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            model.decoder.layers[layer].fc1 = None
            model.decoder.layers[layer].fc2 = None
        else:
            model.decoder.layers[layer].fc1 = prune_linear_layer(model.decoder.layers[layer].fc1, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
            model.decoder.layers[layer].fc2 = prune_linear_layer(model.decoder.layers[layer].fc2, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)


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

def load_pruned_model(model, weights):
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    zs = {}

    architecture = config.architectures[0].lower()
    bert_name = "roberta" if "roberta" in architecture else "bert"

    hidden_z = torch.zeros(config.hidden_size)
    hidden_z[:weights[f"{bert_name}.embeddings.word_embeddings.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{bert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"bert.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z

    prune_model_with_z(zs, model)
    model.load_state_dict(weights, strict=False)
    return model

def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name)
    model_size = calculate_parameters(model)
    return model_size


