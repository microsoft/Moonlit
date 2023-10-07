import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import pandas as pd
import numpy as np
from tqdm import tqdm

gpu_id = 1
device = torch.device(f'cuda:{gpu_id}')

class Decoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, output_dims=6, layers=2):
        super(Decoder, self).__init__()
        self.linear_in = nn.Linear(latent_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_out = nn.Linear(hidden_dims, output_dims)

    def forward(self, z):
        z = self.linear_in(z)
        z = F.relu(z)
        for linear_layer in self.linear_mid:
            z = F.relu(z)
            z = linear_layer(z)
        z = self.linear_out(z)
        z = torch.sigmoid(z)
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=2, hidden_dims=8, input_dims=6, layers=2):
        super(VariationalEncoder, self).__init__()
        self.linear_in = nn.Linear(input_dims, hidden_dims)
        mid_layers = [nn.Linear(hidden_dims, hidden_dims) for _ in range(layers - 2)]
        self.linear_mid = nn.ModuleList(mid_layers)
        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        self.linear_sigma = nn.Linear(hidden_dims, latent_dims)
        self.kl = 0

    def forward(self, x):
        x = self.linear_in(x)
        x = F.relu(x)
        for linear_layer in self.linear_mid:
            x = linear_layer(x)
            x = F.relu(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(
            self, latent_dims=2, source_dims=6,
            encoder_hidden_dims=256, decoder_hidden_dims=256,
            encoder_layers=2, decoder_layers=3
        ):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, encoder_hidden_dims, source_dims, encoder_layers)
        self.decoder = Decoder(latent_dims, decoder_hidden_dims, source_dims, decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



