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
batch_size = 8
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=dataloading.collate_fn)
m, s = dataloading.compute_signal_mean_and_std(device, dataloader)
m = m.detach().cpu()
s = s.detach().cpu()
dataloader.dataset.set_mean_and_std(m, s)
torch.save(dataloader, 'data/dataloader.pth')
print('Dataloader saved')
print('')

# Models, loss
print('Model creation...')
n_conv_channel_1 = config['conv_rec_net']['n_conv_channel_1']
n_conv_channel_2 = config['conv_rec_net']['n_conv_channel_2']
n_conv_channel_3 = config['conv_rec_net']['n_conv_channel_3']
lstm_hidden_size = config['conv_rec_net']['lstm_hidden_size']
n_lstm_layer = config['conv_rec_net']['n_lstm_layer']
model = modeling.ConvRecAutoencoder(3, n_conv_channel_1, n_conv_channel_2,
                                    n_conv_channel_3, lstm_hidden_size,
                                    n_lstm_layer)
model = model.to(device)
print(('The convolutional recurrent model has '
       f'{modeling.count_parameters(model)} parameters'))
loss_fn = modeling.sequence_l1
print('')

# Training
print('Training...')
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
model.train_one_epoch(device, dataloader, loss_fn, optimizer)
torch.save(model.checkpoint, 'scripts/model.pt')
plt.plot(model.checkpoint['loss_history'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('viz/loss.png')
plt.close()
print('Models saved')
print('')

# Encoding
embeddings = model.encode(device, dataloader)
torch.save(embeddings, 'data/embeddings.pt')
print('Embeddings saved')
