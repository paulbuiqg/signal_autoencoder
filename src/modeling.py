"""Autoencoder for 1D multichannel signals of variable-length."""


from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SignalEncoder(nn.Module):
    """Encoder with 1D-convolution and LTSM layers."""

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.conv_enc = nn.Sequential(
            nn.Conv1d(n_channel_in, n_conv_channel_1, 3,
                      stride=1, padding=1),
            nn.BatchNorm1d(n_conv_channel_1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(n_conv_channel_1, n_conv_channel_2, 3,
                      stride=1, padding=1),
            nn.BatchNorm1d(n_conv_channel_2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(n_conv_channel_2, n_conv_channel_3, 3,
                      stride=1, padding=1),
        )
        self.lstm_enc = nn.LSTM(n_conv_channel_3, lstm_hidden_size,
                                num_layers=n_lstm_layer, batch_first=True)

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.permute(0, 2, 1)  # Channel dimension 2nd
        x = self.conv_enc(x)
        x = x.permute(0, 2, 1)  # Feature dimension 2nd
        x, (h, c) = self.lstm_enc(x)
        return x, (h, c)


class SignalDecoder(nn.Module):
    """Decoder with LTSM and 1D-convolution layers.

    Linear layer after LTSM to resize features to adjust channel dimension
    for convolution.
    """

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.lstm_dec = nn.LSTM(lstm_hidden_size, lstm_hidden_size,
                                num_layers=n_lstm_layer, batch_first=True)
        self.linear_dec = nn.Linear(lstm_hidden_size, n_conv_channel_3)
        self.conv_dec = nn.Sequential(
            nn.Conv1d(n_conv_channel_3, n_conv_channel_2,
                      3, stride=1, padding=1),
            nn.BatchNorm1d(n_conv_channel_2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(n_conv_channel_2, n_conv_channel_1,
                      3, stride=1, padding=1),
            nn.BatchNorm1d(n_conv_channel_1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(n_conv_channel_1, n_channel_in,
                      3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, s: Tuple[torch.Tensor, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.lstm_dec(x, s)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)  # Channel dimension 2nd
        x = self.conv_dec(x)
        x = x.permute(0, 2, 1)  # Feature dimension 2nd
        return x


class SignalAutoencoder(nn.Module):
    """Autoencoder with outer LTSM and inner 1D-convolution layers."""

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.encoder = SignalEncoder(n_channel_in, n_conv_channel_1,
                                     n_conv_channel_2, n_conv_channel_3,
                                     lstm_hidden_size, n_lstm_layer)
        self.decoder = SignalDecoder(n_channel_in, n_conv_channel_1,
                                     n_conv_channel_2, n_conv_channel_3,
                                     lstm_hidden_size, n_lstm_layer)
        # # Checkpoint
        # self.checkpoint = {
        #     'train_loss_history': [],
        #     'val_loss_history': [],
        #     'epochs': 0,
        #     'best_val_loss': float('inf'),
        #     'best_epoch': 0,
        #     'patience_counter': 0,
        # }
        # Checkpoint
        self.checkpoint = {
            'loss_history': [],
            'epochs': 0,
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'patience_counter': 0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, c) = self.encoder(x)
        x = self.decoder(x, (h, c))
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input, return the codes."""
        _, (h, _) = self.encoder(x)
        h = h.permute((1, 0, 2))  # Batch dimension 1st
        return torch.flatten(h, start_dim=1)

    def train_one_epoch(
        self,
        device: str,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.optim.Optimizer, list]:
        """Run one training epoch; update optimizer and loss history."""
        self.train()
        loss_history = []
        pbar = tqdm(dataloader, desc='Training', unit='batch')
        for X, le, ma in dataloader:
            X, le, ma = X.to(device), le.to(device), ma.to(device)
            X_pred = self(X)
            loss = loss_fn(X, X_pred, le, ma)
            del X, X_pred, le, ma
            loss.backward()
            loss_item = loss.item()
            del loss
            loss_history.append(loss_item)
            pbar.set_postfix_str(f'loss={loss_item:>8f}')
            optimizer.step()
            optimizer.zero_grad()
        self.update_checkpoint(optimizer, loss_history)

    def update_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        loss_history: list,
    ):
        """Update the checkpoint dictionary."""
        self.checkpoint['model_state_dict'] = self.state_dict()
        self.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        self.checkpoint['train_loss_history'] += loss_history
        self.checkpoint['epochs'] += 1

    # def train_one_epoch(
    #     self,
    #     device: str,
    #     dataloader: DataLoader,
    #     loss_fn: torch.nn.Module,
    #     optimizer: torch.optim.Optimizer,
    # ) -> Tuple[torch.optim.Optimizer, list]:
    #     """Run one training epoch; update optimizer and loss history."""
    #     self.train()
    #     loss_history = []
    #     pbar = tqdm(dataloader, desc='Training', unit='batch')
    #     for X, le, ma in dataloader:
    #         X, le, ma = X.to(device), le.to(device), ma.to(device)
    #         X_pred = self(X)
    #         loss = loss_fn(X, X_pred, le, ma)
    #         del X, X_pred, le, ma
    #         loss.backward()
    #         loss_item = loss.item()
    #         del loss
    #         loss_history.append(loss_item)
    #         pbar.set_postfix_str(f'loss={loss_item:>8f}')
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     return optimizer, loss_history

    # def train_many_epochs(
    #     self,
    #     device: str,
    #     dataloader_train: DataLoader,
    #     dataloader_val: DataLoader,
    #     loss_fn: torch.nn.Module,
    #     optimizer: torch.optim.Optimizer,
    #     n_epoch: int,
    #     patience: int
    # ):
    #     """Run several training epochs; update checkpoint after each epoch."""
    #     epoch = self.checkpoint['epochs'] + 1
    #     for epoch in range(epoch, epoch + n_epoch):
    #         print(f'-- Epoch {epoch} --')
    #         if self.early_stopping(patience):
    #             print('Early stopping!')
    #             break
    #         optimizer, loss_history = self.train_one_epoch(
    #             device, dataloader_train, loss_fn, optimizer)
    #         val_loss = self.evaluate(device, dataloader_val, loss_fn)
    #         print(f'Validation loss: {val_loss:>8f}')
    #         self.update_checkpoint(optimizer, epoch, loss_history, val_loss)

    # def evaluate(
    #     self,
    #     device: str,
    #     dataloader: DataLoader,
    #     loss_fn: torch.nn.Module,
    # ) -> float:
    #     """Compute (mean) loss from labeled data and their prediction."""
    #     self.eval()
    #     pbar = tqdm(dataloader, desc='Evaluation', unit='batch')
    #     mean_loss = 0
    #     with torch.no_grad():
    #         for X, le, ma in pbar:
    #             X, le, ma = X.to(device), le.to(device), ma.to(device)
    #             X_pred = self(X)
    #             loss = loss_fn(X, X_pred, le, ma)
    #             del X, X_pred, le, ma
    #             loss.backward()
    #             del loss
    #             loss_item = loss.item()
    #             mean_loss += loss_item / len(dataloader)
    #             pbar.set_postfix_str(f'loss={loss_item:>8f}')
    #     return mean_loss

    # def update_checkpoint(
    #     self,
    #     optimizer: torch.optim.Optimizer,
    #     epoch: int,
    #     train_loss_history: list,
    #     val_loss: float
    # ):
    #     """Update the checkpoint dictionary."""
    #     self.checkpoint['model_state_dict'] = self.state_dict()
    #     self.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    #     self.checkpoint['train_loss_history'] += train_loss_history
    #     self.checkpoint['val_loss_history'] += [val_loss] * \
    #         len(train_loss_history)
    #     self.checkpoint['epochs'] += 1
    #     if val_loss < self.checkpoint['best_val_loss']:
    #         self.checkpoint['best_model_state_dict'] = self.state_dict()
    #         self.checkpoint['best_val_loss'] = val_loss
    #         self.checkpoint['best_epoch'] = epoch

    # def early_stopping(self, patience: int) -> bool:
    #     """Check if patience limit has been reached."""
    #     if self.checkpoint['epochs'] == 0:
    #         return False
    #     val_loss = self.checkpoint['val_loss_history'][-1]
    #     if self.checkpoint['best_val_loss'] < val_loss:
    #         self.checkpoint['patience_counter'] += 1
    #     else:
    #         self.checkpoint['patience_counter'] = 0
    #     return patience <= self.checkpoint['patience_counter']


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_l1(seq_in: torch.Tensor, seq_out: torch.Tensor,
                lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss between two padded, masked sequences.

    The mask is used to remove the effect of padded values.
    """
    max_len = seq_in.size(1)
    batch_size = seq_in.size(0)
    # Sum of feature dimension
    err = torch.sum(torch.abs(seq_in - seq_out) * mask, 2)
    # Lengths
    lens = lengths.repeat((max_len, 1)).T
    return torch.sum(err / lens / batch_size)
