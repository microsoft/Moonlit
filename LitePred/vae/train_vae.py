import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import pandas as pd
import numpy as np
from vae_model import VariationalAutoencoder 

gpu_id = 1
device = torch.device(f'cuda:{gpu_id}')

def train(autoencoder, data_train, data_val, epochs=100, loss_balance=10, batch_size=16, print_cycle=50):
    autoencoder.to(device)
    iterations = int(np.ceil(10000 / batch_size))
    print(f'Data size: {data_train.shape}, Batch size: {batch_size}, Iteration size: {iterations}')
    opt = torch.optim.Adam(autoencoder.parameters())
    beta = 0.02
    loss_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        np.random.shuffle(data_train)
        sum_loss = np.zeros(2)
        for i in range(iterations):
            x = np.random.choice(data_train.shape[0], p=data_train[:, -1], size=batch_size)
            x = data_train[x, :-1]
            x = torch.from_numpy(x).to(device)
            opt.zero_grad()
            x_hat = autoencoder(x).to(device)
            reconstruction_loss = ((x - x_hat)**2).sum()
            kl_loss = autoencoder.encoder.kl
            loss = reconstruction_loss + kl_loss * beta  
            loss.backward()
            opt.step()
            sum_loss += np.array([reconstruction_loss.item(), kl_loss.item()])
            if i % print_cycle == print_cycle - 1:
                avg_loss = sum_loss / print_cycle / batch_size
                x_val = torch.from_numpy(data_val[:, :-1]).to(device)
                x_hat_val = autoencoder(x_val).to(device)
                reconstruction_loss_val = ((x_val - x_hat_val)**2).sum()
                kl_loss_val = autoencoder.encoder.kl
                sum_loss_val = np.array([reconstruction_loss_val.item(), kl_loss_val.item()])
                avg_loss_val = sum_loss_val / data_val.shape[0]

                loss_list.append([avg_loss, avg_loss_val])
                print(f'Iteration {i + 1}, Beta: {"%.5f" % beta}')
                print(f'Avg Loss Train: {"%.4f" % avg_loss[0]}, {"%.1f" % avg_loss[1]}')
                print(f'Avg Loss Val: {"%.4f" % avg_loss_val[0]}, {"%.1f" % avg_loss_val[1]}')
                beta = 0.1 / (1 + np.exp(2 * (np.log(avg_loss[0]) + loss_balance)))
                sum_loss = 0

    return autoencoder, loss_list

if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
        
    features = ['HW', 'CIN', 'COUT', 'KERNEL_SIZE', 'STRIDES']
    lut_configs = pd.read_csv('./configs/vae_inputs_patch_lut.csv')
    lut_configs_train = lut_configs[lut_configs['CAT'] == 'train'][features + ['SCALED_FREQ']]
    lut_configs_test = lut_configs[lut_configs['CAT'] == 'test'][features + ['SCALED_FREQ']]
    df = lut_configs[features].copy()
    df['HW'] = np.log2(df['HW'])
    df['CIN'] = np.log2(df['CIN'])
    df['COUT'] = np.log2(df['COUT'])
    data = df.to_numpy().astype(np.float32)
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    data = (data - min_val) / (max_val - min_val)
    scale = max_val - min_val
    data = np.hstack((data, lut_configs[['SCALED_FREQ']].astype(np.float32)))
    data_train = data[lut_configs['CAT'] == 'train']
    data_val = data[lut_configs['CAT'] == 'test']
    from train_vae import VariationalAutoencoder as VAE
    vae_path = './models/test/vae.model'

    latent_dims = 128
    encoder_hidden_dims = 256
    decoder_hidden_dims = 256
    source_dims = 5
    encoder_layers = 5
    decoder_layers = 5
    model = VariationalAutoencoder(
        latent_dims=latent_dims,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        source_dims=source_dims,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )

    vae, loss = train(model, data_train, data_val, epochs=1, loss_balance=8)

    torch.save(model.state_dict(), f'./checkpoint/model.pth')