"""Train the autoencoder, save the trained model, compute embeddings."""


import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from signal_autoencoder import dataloading
from signal_autoencoder import modeling


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

# Model, loss, optimizer
print('Model creation...')
n_conv_channel_1 = 64
n_conv_channel_2 = 128
n_conv_channel_3 = 256
lstm_hidden_size = 256
n_lstm_layer = 1
model = modeling.SignalAutoencoder(3, n_conv_channel_1, n_conv_channel_2,
                                n_conv_channel_3, lstm_hidden_size,
                                n_lstm_layer).to(device)
print(f'The model has {modeling.count_parameters(model)} parameters')
loss_fn = modeling.sequence_l1
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
print('')

# Training
print('Training...')
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/autoencoder.pt')):
    model.train_one_epoch(device, dataloader, loss_fn, optimizer)
    torch.save(model.checkpoint, 'scripts/autoencoder.pt')
checkpoint = torch.load('scripts/autoencoder.pt')
plt.plot(checkpoint['loss_history'], label='Training')
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig('viz/loss.png')
print('Model saved')
print('')

# Encoding
print('Encoding...')
if not (os.path.isfile('data/dataloader.pth')
        and os.path.isfile('scripts/autoencoder.pt')
        and os.path.isfile('scripts/embeddings.pt')):
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings = model.encode(device, dataloader)
    torch.save(embeddings, 'data/embeddings.pt')
print('Embeddings saved')
