"""Train the autoencoder, save the trained model, compute embeddings."""


import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from signal_autoencoder import dataloading
from signal_autoencoder import modeling


# Config
with open('scripts/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Dataset
# One item: 3-channel signal of seismic waveforms from one event ("trace")
print('Dataset creation...')
events = pd.read_csv('data/events.csv')
dataset = dataloading.SeismicSignals('data', events)
print(f'{len(events)} seismic events')
print(f'{len(dataset)} items')
print('')

# Dataloader
print('Dataloader creation...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
if not os.path.isfile('data/dataloader.pth'):
    batch_size = 4
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataloading.collate_fn)
    m, s = dataloading.compute_signal_mean_and_std(device, dataloader)
    dataloader.dataset.set_mean_and_std(m, s)
    torch.save(dataloader, 'data/dataloader.pth')
dataloader = torch.load('data/dataloader.pth')
print('Dataloader saved')
print('')

# Models, loss
print('Model creation...')
n_conv_channel_1 = config['conv_net']['n_conv_channel_1']
n_conv_channel_2 = config['conv_net']['n_conv_channel_2']
n_conv_channel_3 = config['conv_net']['n_conv_channel_3']
n_conv_channel_4 = config['conv_net']['n_conv_channel_4']
conv_net = modeling.ConvAutoencoder(3, n_conv_channel_1, n_conv_channel_2,
                                    n_conv_channel_3, n_conv_channel_4)
conv_net = conv_net.to(device)
print(('The convolutional model has '
       f'{modeling.count_parameters(conv_net)} parameters'))
#
n_conv_channel_1 = config['conv_rec_net']['n_conv_channel_1']
n_conv_channel_2 = config['conv_rec_net']['n_conv_channel_2']
n_conv_channel_3 = config['conv_rec_net']['n_conv_channel_3']
lstm_hidden_size = config['conv_rec_net']['lstm_hidden_size']
n_lstm_layer = config['conv_rec_net']['n_lstm_layer']
conv_rec_net = modeling.ConvRecAutoencoder(3, n_conv_channel_1,
                                           n_conv_channel_2, n_conv_channel_3,
                                           lstm_hidden_size, n_lstm_layer)
conv_rec_net = conv_rec_net.to(device)
print(('The convolutional recurrent model has '
       f'{modeling.count_parameters(conv_rec_net)} parameters'))
#
loss_fn = modeling.sequence_l1
print('')

# Training
print('Training convolutional neural network...')
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/conv_net.pt')):
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=.001)
    conv_net.train_one_epoch(device, dataloader, loss_fn, optimizer)
    torch.save(conv_net.checkpoint, 'scripts/conv_net.pt')
checkpoint = torch.load('scripts/conv_net.pt')
plt.plot(checkpoint['loss_history'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('viz/conv_net_loss.png')
#
print('Training convolutional recurrent neural network...')
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/conv_rec_net.pt')):
    optimizer = torch.optim.Adam(conv_rec_net.parameters(), lr=.001)
    conv_rec_net.train_one_epoch(device, dataloader, loss_fn, optimizer)
    torch.save(conv_rec_net.checkpoint, 'scripts/conv_rec_net.pt')
checkpoint = torch.load('scripts/conv_rec_net.pt')
plt.plot(checkpoint['loss_history'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('viz/conv_rec_net_loss.png')
#
print('Models saved')
print('')

# Encoding
print('Encoding...')
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/conv_net.pt')
        and os.path.isfile('data/conv_net_embeddings.pt')):
    conv_net.load_state_dict(checkpoint['model_state_dict'])
    embeddings = conv_net.encode(device, dataloader)
    torch.save(embeddings, 'data/conv_net_embeddings.pt')
#
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/conv_rec_net.pt')
        and os.path.isfile('data/conv_rec_net_embeddings.pt')):
    conv_rec_net.load_state_dict(checkpoint['model_state_dict'])
    embeddings = conv_rec_net.encode(device, dataloader)
    torch.save(embeddings, 'data/conv_rec_net_embeddings.pt')
#
print('Embeddings saved')
