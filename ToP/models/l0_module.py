# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pdb

from re import L
from black import main
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)

AVERAGE_SAMPLE_LENGTH = {
    "cola": 11,
    "rte":  64,
    "qqp":  30,
    "mrpc": 53,
    "sst2": 25,
    "mnli": 39,
    "wnli": 37,
    "qnli": 51,
    "stsb": 31,
    "20news": 551,
    "imdb": 264,
    "squad_v2": 152,
    None: 152,
    "yelp": 179,
}

MAX_SEQUENCE_LENGTH = {
    "cola": 64,
    "rte":  256,
    "qqp":  128,
    "mrpc": 128,
    "sst2": 64,
    "mnli": 128,
    "wnli": 128,
    "qnli": 128,
    "stsb": 64,
    "20news": 512,
    "imdb": 512,
    "squad_v2": 384,
    None: 384,
    "yelp": 512,
}

class L0Module(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+head_layer+mlp_layer+token",
                 magical_number=0.8, # from Wang et al. 2020
                 token_prune_loc=[1,2,3,4,5,6,7,8,9,10,11],
                 bin_num=50,
                 ):
        super(L0Module, self).__init__()
        self.bin_num = bin_num
        self.all_types = []
        if "structured_heads" in pruning_type:
            self.all_types.append("head_z")
        if "structured_mlp" in pruning_type:
            self.all_types.append("intermediate_z")
        if "head_layer" in pruning_type:
            self.all_types.append("head_layer_z")
        if "mlp_layer" in pruning_type:
            self.all_types.append("mlp_z")
        if "hidden" in pruning_type:
            self.all_types.append("hidden_z")
        if "token" in pruning_type:
            self.all_types.append("token_z")
        if "pruner" in pruning_type:
            self.all_types.append("pruner_z")
        self.pruning_type = pruning_type

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        self.num_attention_heads = config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.token_prune_loc = token_prune_loc
        self.num_labels = config.num_labels

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads


        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0 

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        types = self.pruning_type.split("+")
        for type in types:
            if type != "layer":
                self.initialize_one_module(type)
        if "layer" in types:
            self.initialize_one_module("layer")

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_3 = torch.tensor(0.0)

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape", self.z_logas[type].shape)
            logger.info(f"size", self.sizes[type])
        logger.info(f"prunable model size: {self.prunable_model_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "head_layer":
            self.initialized_layer_structured_heads()
        elif module_name == "mlp_layer":
            self.initialize_whole_mlp()
        elif module_name == "token":
            self.initialize_token()
        elif module_name == "pruner":
            self.initialize_pruner()
        else:
            print("Not implemented module name:", module_name)
            raise NotImplementedError

    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_head(self, add_prunable_model_size=True):
        self.head_loga = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
        self.reset_loga(self.head_loga, mean=10)
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.num_hidden_layers, 1, self.num_attention_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialized_layer_structured_heads(self):
        n_layer = self.num_hidden_layers
        self.headlayer_loga = self.initialize_parameters(n_layer)
        self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="head_layer", 
                            parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                            shape=[n_layer])
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_mlp(self):
        self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)

        self.add_one_module(self.int_loga, type="intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        self.reset_loga(self.int_loga)
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")

    def initialize_whole_mlp(self):
        n_layer = self.num_hidden_layers
        self.intlayer_loga = self.initialize_parameters(n_layer)
        self.add_one_module(self.intlayer_loga, type="mlp", 
                            parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                            shape=[n_layer])
        self.reset_loga(self.intlayer_loga, mean=10)
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")

    def initialize_token(self):
        rank_bin_num = self.bin_num
        self.token_loga = self.initialize_parameters(rank_bin_num, len(self.token_prune_loc))
        self.add_one_module(self.token_loga, type="token",
                            parameter_per_dim=None, size=rank_bin_num,
                            shape=[len(self.token_prune_loc), rank_bin_num])
        self.reset_loga(self.token_loga, mean=10)
        logger.info(f"Initialized token!")
    
    def initialize_pruner(self):
        n_layer = len(self.token_prune_loc) + 1
        self.pruner_loga = self.initialize_parameters(n_layer - 1)
        self.add_one_module(self.pruner_loga, type="pruner",
                            parameter_per_dim=None, size=1,
                            shape=[n_layer - 1])
        self.reset_loga(self.pruner_loga, mean=-10)
        logger.info(f"Initialized pruner!")
        
    def reset_loga(self, tensor, mean=None, droprate_init=None):
        if droprate_init is None:
            droprate_init = self.droprate_init
        if mean is None:
            mean = math.log(1 - droprate_init) - math.log(droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score

    def get_num_parameters_for_mlp(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current sparsity
        num_parameters = 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        return num_parameters

    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        # all_head_score: 12,1,1
        # head_score: 12,12,1
        all_head_score, head_score = self.transform_scores_for_head()
        
        head_score = head_score * all_head_score
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        if "mlp" in self.types:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
            intlayer_score = intlayer_score.unsqueeze(-1)
        else:
            intlayer_score = 1.0
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072

        int_score = int_score * intlayer_score
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity

    def get_warmup_progress(self, pruned_steps):
        return min(1, pruned_steps / self.lagrangian_warmup)

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        soft_mask = torch.where(
            soft_mask > 0.0,
            torch.ones_like(soft_mask, device=soft_mask.device),
            torch.zeros_like(soft_mask, device=soft_mask.device),
        )
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            if self.disable_token_pruning(type):
                continue
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                new_z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = new_z
        return numpified_zs

    def disable_token_pruning(self, type="token"):
        # return self.lagrangian_warmup > 0 and type.startswith("token") and len(self.types) != 1
        return False

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}
        if self.disable_token_pruning() and "token_z" in zs:
            del zs["token_z"]

        if training:
            for i, type in enumerate(self.types):
                if self.disable_token_pruning(type):
                    continue
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if self.disable_token_pruning(type):
                    continue
                if type != "hidden": # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])
        return zs


class L0ModuleForMAC(L0Module):
    # Reference:
    # MAC calculation: https://zhuanlan.zhihu.com/p/463177605
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 token_prune_loc=[3, 6, 9],
                 bin_num=50,
    ):
        super().__init__(
            config=config,
            droprate_init=droprate_init,
            temperature=temperature,
            lagrangian_warmup=lagrangian_warmup,
            start_sparsity=start_sparsity,
            target_sparsity=target_sparsity,
            pruning_type=pruning_type,
            magical_number=magical_number,
            token_prune_loc=token_prune_loc,
            bin_num=bin_num,
        )
        self.config = config

    def calculate_mac_for_one_attention_head(self, seq_len, hidden_size, hidden_size_per_head):
        qkv_mac = 3 * seq_len * hidden_size * hidden_size_per_head
        qk_mac = seq_len * hidden_size_per_head * seq_len
        softmax_mac = seq_len * seq_len
        v_mac = seq_len * hidden_size_per_head * seq_len
        return qkv_mac + qk_mac + softmax_mac + v_mac
    
    def calculate_mac_for_one_mha(self, seq_len, hidden_size, num_attention_heads, hidden_size_per_head):
        mac = num_attention_heads * self.calculate_mac_for_one_attention_head(seq_len, hidden_size, hidden_size_per_head)
        # dense mac
        mac += seq_len * hidden_size * hidden_size
        return mac
    
    def calculate_mac_for_one_ffn(self, seq_len, hidden_size, intermediate_size):
        # dense mac
        mac = seq_len * hidden_size * intermediate_size
        # dense mac
        mac += seq_len * intermediate_size * hidden_size
        return mac

    def calculate_mac_for_one_layer(
        self,
        att_seq_len, ffn_seq_len,
        hidden_size, intermediate_size, num_attention_heads, hidden_size_per_head,
    ):
        mac = 0
        mac += self.calculate_mac_for_one_mha(
            att_seq_len, hidden_size, num_attention_heads[0], hidden_size_per_head,
        )
        mac += self.calculate_mac_for_one_ffn(
            ffn_seq_len, hidden_size, intermediate_size,
        )
        return mac

    def calculate_mac_for_encoder(
        self, token_length_for_each_layer, hidden_size,
        intermediate_size_list, num_attention_heads_list, 
        num_hidden_layers, hidden_size_per_head,
    ):
        mac = 0
        for i in range(num_hidden_layers):
            mac += self.calculate_mac_for_one_layer(
                att_seq_len=token_length_for_each_layer[i],
                ffn_seq_len=token_length_for_each_layer[min(i+1, num_hidden_layers-1)],
                hidden_size=hidden_size,
                intermediate_size=intermediate_size_list[i],
                num_attention_heads=num_attention_heads_list[i],
                hidden_size_per_head=hidden_size_per_head,
            )
        return mac
    
    def calculate_mac_for_model(
        self,
        token_length_for_each_layer = [128] * 12,
        hidden_size = 768,
        intermediate_size_list = [3072] * 12,
        num_attention_heads_list = [[12]] * 12,
        hidden_size_per_head = 64,
        num_hidden_layers = 12,
        num_labels = 2,
        vocab_size = 30522,
    ):
        intermediate_size_list = intermediate_size_list[:num_hidden_layers]
        num_attention_heads_list = num_attention_heads_list[:num_hidden_layers]
        assert len(token_length_for_each_layer) == num_hidden_layers
        assert len(intermediate_size_list) == num_hidden_layers
        assert len(num_attention_heads_list) == num_hidden_layers
        mac = self.calculate_mac_for_encoder(
            token_length_for_each_layer=token_length_for_each_layer,
            hidden_size=hidden_size,
            intermediate_size_list=intermediate_size_list,
            num_attention_heads_list=num_attention_heads_list,
            num_hidden_layers=num_hidden_layers,
            hidden_size_per_head=hidden_size_per_head,
        )
        return mac
    
    def calculate_mac_for_one_layer_vectorized(
        self,
        att_seq_len: torch.tensor,
        ffn_seq_len: torch.tensor,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12, 
        hidden_size_per_head: int = 64,
    ):
        assert att_seq_len.shape == ffn_seq_len.shape
        mac = 0.0
        mac += self.calculate_mac_for_one_mha(
            att_seq_len, hidden_size, num_attention_heads, hidden_size_per_head,
        )
        mac += self.calculate_mac_for_one_ffn(
            ffn_seq_len, hidden_size, intermediate_size,
        )
        return mac
    
    def calculate_mac_for_encoder_vectorized(
        self, token_length_for_each_layer: torch.tensor,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        hidden_size_per_head: int = 64,
        num_hidden_layers: int = 12,
    ):
        mac = self.calculate_mac_for_one_layer_vectorized(
            att_seq_len=token_length_for_each_layer,
            ffn_seq_len=torch.hstack([token_length_for_each_layer[..., 1:], token_length_for_each_layer[..., -1:]]),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            hidden_size_per_head=hidden_size_per_head,
        ).sum(-1)
        return mac
    
    def calculate_mac_for_model_vectorized(
        self, token_length_for_each_layer: torch.tensor,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        hidden_size_per_head: int = 64,
        num_hidden_layers: int = 12,
        num_labels: int = 2,
        vocab_size: int = 30522,
    ):
        assert len(token_length_for_each_layer.shape) == 2 and token_length_for_each_layer.shape[1] == num_hidden_layers
        mac = self.calculate_mac_for_encoder_vectorized(
            token_length_for_each_layer=token_length_for_each_layer,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            hidden_size_per_head=hidden_size_per_head,
            num_hidden_layers=num_hidden_layers,
        )
        return mac

    def get_mac_and_constraint(
        self,
        manually_add_CLS_token=True,
        disable_token_pruning=False,
        attention_mask=None,
        token_score=None,
        pruner_score=None,
    ):
        assert manually_add_CLS_token
        assert not disable_token_pruning
        # assert self.num_hidden_layers == 12
        assert len(self.types) == 2 and self.types[0] == "token" and self.types[1] == "pruner"

        if "token" in self.types:
            if token_score is None:
                token_score = 1 - self.cdf_qz(0, self.token_loga)  # 12 * 128
            if pruner_score is None:
                pruner_score = 1 - self.cdf_qz(0, self.pruner_loga)  # 11
            token_length_for_each_layer = torch.mean(token_score, dim=1)

            # add this line.
            # if pruner_score is 1, it means prune in this layer.
            # if pruner_score is 0, it means do not prune in this layer.
            token_length_for_each_layer = 1.0 - ((1.0 - token_length_for_each_layer) * pruner_score)

            token_length_for_each_layer_list = []
            token_length_for_each_layer_list.append(torch.tensor(1.0, device=token_length_for_each_layer.device))
            p_count = 0
            for i in range(1, self.num_hidden_layers):
                if i in self.token_prune_loc:
                    token_length_for_each_layer_list.append(
                        token_length_for_each_layer_list[-1] * token_length_for_each_layer[p_count]
                    )
                    p_count += 1
                else:
                    token_length_for_each_layer_list.append(token_length_for_each_layer_list[-1])
            assert p_count == len(self.token_prune_loc)
            full_lengths = attention_mask.sum(-1).float()
            # vectorized version
            B = full_lengths.shape[0]
            full_lengths = full_lengths.unsqueeze(-1).repeat(1, self.num_hidden_layers)
            token_length_for_each_layer_list = torch.stack(token_length_for_each_layer_list).unsqueeze(0).repeat(B, 1)
            pruned_lengths = (full_lengths - 1) * token_length_for_each_layer_list + 1
            macs = self.calculate_mac_for_model_vectorized(
                token_length_for_each_layer=pruned_lengths,
                num_labels=self.num_labels,
                num_hidden_layers=self.num_hidden_layers,
            )
            full_macs = self.calculate_mac_for_model_vectorized(
                token_length_for_each_layer=full_lengths,
                num_labels=self.num_labels,
                num_hidden_layers=self.num_hidden_layers,
            )
            relative_macs = (macs / full_macs).mean()

            task = self.config.finetuning_task
            if task is None:
                task = 'squad_v2'
            max_sequence_lengths = torch.ones_like(full_lengths) * MAX_SEQUENCE_LENGTH[task]
            sequence_full_macs = self.calculate_mac_for_model_vectorized(
                token_length_for_each_layer=max_sequence_lengths,
                num_labels=self.num_labels,
                num_hidden_layers=self.num_hidden_layers,
            )
            relative_sequence_macs = (macs / sequence_full_macs).mean()
            return relative_macs, relative_sequence_macs
        else:
            raise NotImplementedError

    def lagrangian_regularization(
        self,
        pruned_steps,
        attention_mask=None,
        token_score=None,
        pruner_score=None,
    ):
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        else:
            target_sparsity = self.target_sparsity

        expected_relative_mac, expected_sequence_relative_mac = self.get_mac_and_constraint(
            attention_mask=attention_mask,
            token_score=token_score,
            pruner_score=pruner_score,
        )

        expected_sparsity = 1 - expected_relative_mac
        expected_sequence_sparsity = 1 - expected_sequence_relative_mac
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = ( #! see appendix
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
        )
        return lagrangian_loss, expected_sparsity, target_sparsity, expected_sequence_sparsity

    def get_mask_lambda(self, pruned_steps):
        self.lambda_3 = self.lambda_2 * 0.001 * min(1, pruned_steps / self.lagrangian_warmup)
        return self.lambda_3

    def calculate_model_size(self, zs, task_name=None):
        numpified_zs = self.get_z_from_zs(zs)
        token_z = numpified_zs["token"].astype(bool)
        pruner_z = numpified_zs["pruner"].astype(bool)
        assert token_z.shape[0] == len(self.token_prune_loc) and len(token_z.shape) == 2
        rank_bin_num = token_z.shape[1]

        temp = token_z.sum(-1).astype(float) / rank_bin_num
        remaining_token_ratio = np.ones(self.num_hidden_layers)
        for i, loc in enumerate(self.token_prune_loc):
            if pruner_z[i]:
                remaining_token_ratio[loc] = temp[i]
        remaining_token_ratio = np.round(np.cumprod(remaining_token_ratio), 2).tolist()


        standard_seq_length = AVERAGE_SAMPLE_LENGTH[task_name]
        remaining_token_nums = [int((AVERAGE_SAMPLE_LENGTH[task_name] - 1) * r) + 1 for r in remaining_token_ratio]
        remaining_macs = self.calculate_mac_for_model(
            token_length_for_each_layer=remaining_token_nums,
            num_labels=self.num_labels,
            num_hidden_layers=self.num_hidden_layers,
        )
        full_macs = self.calculate_mac_for_model(
            token_length_for_each_layer=[standard_seq_length] * self.num_hidden_layers,
            num_labels=self.num_labels,
            num_hidden_layers=self.num_hidden_layers,
        )

        token_score = 1 - self.cdf_qz(0, self.token_loga)  # 12 * 128
        pruner_score = 1 - self.cdf_qz(0, self.pruner_loga)  # 12
        token_ratio_for_training = torch.mean(token_score, dim=1)
        # add this line. 
        # if pruner_score is 1, it means prune in this layer.
        # if pruner_score is 0, it means do not prune in this layer.
        token_ratio_for_training = 1.0 - ((1.0 - token_ratio_for_training) * pruner_score)

        token_ratio_for_inference = 1 - ((1 - temp) * pruner_z)
        results = {}
        results["token_z"] = np.where(np.expand_dims(pruner_z, -1), token_z, np.ones_like(token_z))
        results["token_ratio_for_training"] = token_ratio_for_training
        results["token_ratio_for_inference"] = np.round(token_ratio_for_inference, 2).tolist()
        results["token_prune_loc"] = pruner_z.tolist()
        results["remaining_macs"] = remaining_macs
        results["remaining_token_ratio"] = remaining_token_ratio
        results["macs_sparsity"] = round(1 - remaining_macs / full_macs, 4)
        results["lambda_1"] = self.lambda_1.item()
        results["lambda_2"] = self.lambda_2.item()
        results["lambda_3"] = self.lambda_3.item()
        return results

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("bert-base-uncased")
    # l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
    l0_module = L0ModuleForMAC(config, pruning_type="token")

    import numpy as np
    # task = "sst2"
    # text = "0 0.977074146270752 1 0.976719081401825 2 0.9411757588386536 3 0.9399062991142273 4 0.9395936131477356 5 0.9394234418869019 6 0.9390215873718262 7 0.9389373660087585 8 0.9388896822929382 9 0.937352180480957 10 0.568627119064331 11 0.4232681393623352"
    # text = "0 0.9665433764457703 1 0.9651890397071838 2 0.8044235110282898 3 0.8030996918678284 4 0.8025591969490051 5 0.8024323582649231 6 0.8016939759254456 7 0.6838764548301697 8 0.6837866902351379 9 0.6779482960700989 10 0.3207396864891052 11 0.08739133179187775"
    # text = "0 0.9621531963348389 1 0.9612972140312195 2 0.7635027170181274 3 0.7500730156898499 4 0.7491877675056458 5 0.7467833161354065 6 0.744597852230072 7 0.49801504611968994 8 0.4978805184364319 9 0.4968653619289398 10 0.17740707099437714 11 0.061266832053661346"
    # text = "0 0.9594794511795044 1 0.958853006362915 2 0.9083590507507324 3 0.6870470643043518 4 0.6859422922134399 5 0.5170767307281494 6 0.5133189558982849 7 0.4330674111843109 8 0.43293285369873047 9 0.4308219850063324 10 0.09527174383401871 11 0.04484688118100166"
    # text = "0 0.9591585397720337 1 0.9578106999397278 2 0.8290846347808838 3 0.6650428771972656 4 0.6648193597793579 5 0.4770842492580414 6 0.46519675850868225 7 0.36063307523727417 8 0.3604985177516937 9 0.3594006896018982 10 0.06913332641124725 11 0.04463919624686241"
    # text = "0 0.9578774571418762 1 0.9458622932434082 2 0.5947229266166687 3 0.4665765166282654 4 0.46604371070861816 5 0.2984601557254791 6 0.2524665296077728 7 0.2093750536441803 8 0.20916242897510529 9 0.17856210470199585 10 0.04095212742686272 11 0.04041578248143196"
    
    # task = 'mnli'
    # text = "0 0.9998973608016968 1 0.9997435212135315 2 0.9318457841873169 3 0.9316579699516296 4 0.841589093208313 5 0.8145332932472229 6 0.8144321441650391 7 0.5966497659683228 8 0.5965566039085388 9 0.5964862108230591 10 0.3629887104034424 11 0.316977322101593"
    # text = "0 0.9979664087295532 1 0.955072820186615 2 0.8909671306610107 3 0.8887819051742554 4 0.7377876043319702 5 0.7141939997673035 6 0.6701866984367371 7 0.483471542596817 8 0.4487602710723877 9 0.44871217012405396 10 0.28451353311538696 11 0.23931531608104706"
    # text = "0 0.9893448352813721 1 0.9429304003715515 2 0.8610376119613647 3 0.8088688254356384 4 0.6706724762916565 5 0.625174343585968 6 0.5651005506515503 7 0.3925708532333374 8 0.362978994846344 9 0.36162692308425903 10 0.23281028866767883 11 0.19626706838607788"
    # text = "0 0.9756559133529663 1 0.8253880143165588 2 0.7477332949638367 3 0.6710875630378723 4 0.5647487044334412 5 0.5016770362854004 6 0.3918144702911377 7 0.27823132276535034 8 0.23278909921646118 9 0.2187485545873642 10 0.159937784075737 11 0.14147324860095978"
    # text = "0 0.9743920564651489 1 0.7307940125465393 2 0.6513301730155945 3 0.5572355389595032 4 0.4729238748550415 5 0.4093806743621826 6 0.29166853427886963 7 0.21047404408454895 8 0.1719425916671753 9 0.15728595852851868 10 0.12130168080329895 11 0.11474072188138962"

    task = '20news'
    text = "0 0.989919126033783 1 0.8268734812736511 2 0.8266393542289734 3 0.8262272477149963 4 0.5226477384567261 5 0.5164554715156555 6 0.2553824186325073 7 0.25535908341407776 8 0.15047965943813324 9 0.13554447889328003 10 0.10634975135326385 11 0.10634137690067291"

    text = text.split(" ")
    text = [float(t) for t in text]
    text = text[1::2]
    ratio = np.array(text)
    lengths = ratio * AVERAGE_SAMPLE_LENGTH[task]
    pruned_mac = 0
    for length in lengths:
        layer_mac = l0_module.calculate_mac_for_one_layer(
            att_seq_len=length,
            ffn_seq_len=length,
            hidden_size=768,
            num_attention_heads=[12],
            intermediate_size=3072,
            hidden_size_per_head=64,
        )
        pruned_mac += layer_mac
    full_mac = l0_module.calculate_mac_for_model(token_length_for_each_layer=[AVERAGE_SAMPLE_LENGTH[task]] * 12)
    seq_full_mac = l0_module.calculate_mac_for_model(token_length_for_each_layer=[MAX_SEQUENCE_LENGTH[task]] * 12)
    print("sparsity:", pruned_mac, full_mac, round(1 - pruned_mac / full_mac, 4))
    print("seq sparsity:", pruned_mac, seq_full_mac, round(1 - pruned_mac / seq_full_mac, 4))

    # import matplotlib.pyplot as plt
    # lengths = np.arange(16, 4096)
    # macs = []
    # for length in lengths:
    #     mac = l0_module.calculate_mac_for_model(token_length_for_each_layer=[length] * 12)
    #     macs.append(mac / 1e9)
    # plt.plot(lengths, macs)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel("Short Token Length --> Long Token Length")
    # plt.ylabel("Low FLOPs (latency) --> High FLOPs (latency)")
    # plt.title("FLOPs (latency) vs Token Length")
    # plt.savefig("mac.png")

