"""Train the autoencoder, save the trained model."""

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import dataloading
import modeling

# %%
# Dataset
# One item: 3-channel signal of seismic waveforms from one event ("trace")
print('Dataset creation...')
try:
    events = pd.read_csv('data/events.csv')
    dataset = dataloading.SeismicSignals('data', events)
except FileNotFoundError:  # For interactive mode in VSCode
    events = pd.read_csv('../data/events.csv')
    dataset = dataloading.SeismicSignals('../data', events)
print(f'{len(events)} seismic events')
print(f'{len(dataset)} items')
print('')

# %%
# Dataloader
print('Dataloader creation...')
batch_size = 4
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                       collate_fn=dataloading.collate_fn)
m, s = dataloading.compute_signal_mean_and_std('cuda', dataloader)
dataloader.dataset.set_mean_and_std(m, s)
print('')

# %%
# Model, loss, optimizer
print('Model creation...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# %%
# Training
print('Training...')
model.train_one_epoch(device, dataloader, loss_fn, optimizer)
plt.plot(np.log(model.checkpoint['loss_history']), label='Training')
plt.xlabel('Iteration')
plt.ylabel('Log-loss')
plt.legend()
try:
    plt.savefig('viz/loss.png')
    torch.save(model.checkpoint, 'src/autoencoder.pt')
except FileNotFoundError:  # For interactive mode in VSCode
    plt.savefig('../viz/loss.png')
    torch.save(model.checkpoint, '../src/autoencoder.pt')
print('Model saved')

# %%