# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from vae.vae_model import VariationalAutoencoder as VAE
import numpy as np
import pandas as pd

def load_vae_model(weight_path):
    latent_dims = 128
    encoder_hidden_dims = 256
    decoder_hidden_dims = 256
    source_dims = 5
    encoder_layers = 5
    decoder_layers = 5
    model0 = VAE(
        latent_dims=latent_dims,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        source_dims=source_dims,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    model0.load_state_dict(torch.load(weight_path,map_location='cpu'))
    model0 = model0.cpu()
    model0.eval()
    return model0.decoder


def G(model,z,scale,min_val):
    input_z = torch.from_numpy(z).to(torch.float32)
    output_x = model.forward(input_z).detach().numpy()
    output_x[output_x > 1] = 1
    output_x[output_x < 0] = 0
    return output_x * scale + min_val

def R(x):
    x0 = list(x.flatten())
    x0 = [np.exp2(x0[0]), np.exp2(x0[1]), np.exp2(x0[2]), x0[3], x0[4]]
    x0 = [round(val) for val in x0]
    return x0


def generate_data(model,scale,min_val,features,num=2000,latent_dims=128):
    z = np.random.normal(size=(num, latent_dims))
    x_hat_0 = [R(G(model, row,scale,min_val)) for row in z]
    df = pd.DataFrame(data=np.array(x_hat_0), columns=features)
    df['SCALED_FREQ'] = 1 / num
    return df


def reconstraint_data(x: pd.DataFrame):
    x.loc[x['KERNEL_SIZE'] < 3, 'KERNEL_SIZE'] = 3
    x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
    x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
    x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
    x.loc[x['CIN'] > 3, 'CIN'] = \
        np.ceil(x[x['CIN'] > 3]['CIN'] / 8).astype(np.int32) * 8
    x['COUT'] = x['CIN']
    return x


def get_distribution(data, kernel_size, strides):
    window = torch.from_numpy(np.ones((1, 1, 11, 11))) * (5 * 5 / 11 / 11)
    padding = (5, 5)
    stride = (5, 5)
    
    
    data_size = len(data)
    data = data[(data['KERNEL_SIZE'] == kernel_size) & (data['STRIDES'] == strides)]
    freqs = data['SCALED_FREQ']
    data = data[['HW', 'CIN']]
    data['HW'] = np.ceil(np.log2(data['HW'] + 1) * 5)
    data['CIN'] = np.ceil(np.log2(data['CIN'] + 1))
    data = data.to_numpy().astype(np.int32)
    x = np.zeros((49, 49))
    for row, freq in zip(data, freqs):
        x[tuple(row)] += freq
    x = torch.from_numpy(x.reshape([1, 1] + list(x.shape)))
    x = torch.nn.functional.conv2d(x, window, padding=padding, stride=stride)
    x += 1e-10
    return x, freqs.sum()


def calc_divergence(data1, data2):
    divergence = 0
    kernel_size_steps = [1, 2, 3, 4, 5, 6, 7]
    strides_steps = [1, 2]
    for kernel_size in kernel_size_steps:
        for strides in strides_steps:
            p, p_freq = get_distribution(data1, kernel_size, strides)
            q, q_freq = get_distribution(data2, kernel_size, strides)
            divergence += torch.sum(p * torch.log(p / q)).item() * p_freq
    return divergence


def calc_self_divergence(data):
    data1 = data[:len(data) // 2]
    data2 = data[len(data) // 2:]
    return calc_divergence(data1, data2)