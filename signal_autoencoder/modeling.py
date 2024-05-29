"""Autoencoder for 1D multichannel signals of variable-length."""


from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def conv_block(n_channel_in: int, n_channel_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(n_channel_in, n_channel_out, 3, stride=1, padding=1),
        nn.BatchNorm1d(n_channel_out),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Conv1d(n_channel_out, n_channel_out, 3, stride=1, padding=1),
        nn.BatchNorm1d(n_channel_out)
    )


###############
# Autoencoder #
###############


class Autoencoder(nn.Module):
    """Generic autoencoder class."""

    def __init__(self):
        super().__init__()
        # Checkpoint
        self.checkpoint = {
            'loss_history': [],
            'epochs': 0
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        pass

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
        for X, le, ma in pbar:
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
        self.checkpoint['loss_history'] += loss_history
        self.checkpoint['epochs'] += 1

    def encode(self, device: str, dataloader: DataLoader) -> torch.Tensor:
        """Apply encoder and return embeddings."""
        self.eval()
        pbar = tqdm(dataloader, desc='Encoding', unit='batch')
        embeddings = []
        with torch.no_grad():
            for X, _, _ in pbar:
                X = X.to(device)
                emb = self.embed(X)
                emb = torch.flatten(emb, start_dim=1)
                embeddings.append(emb)
        return torch.cat(embeddings)


############################
# Convolutional neural net #
############################


class ConvEncoder(nn.Module):
    """Encoder with 1D-convolution layers."""

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 n_conv_channel_4: int):
        super().__init__()
        self.enc = conv_block(n_channel_in, n_conv_channel_1)
        self.enc += conv_block(n_conv_channel_1, n_conv_channel_2)
        self.enc += conv_block(n_conv_channel_2, n_conv_channel_3)
        self.enc += conv_block(n_conv_channel_3, n_conv_channel_4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input size: batch, length, channels
        x = x.permute(0, 2, 1)  # Size: batch, channels, length
        x = self.enc(x)
        x = torch.sum(x, 2)  # Size: batch, channels
        return x


class ConvDecoder(nn.Module):
    """Decoder with 1D-convolution layers."""

    def __init__(self, n_channel_out: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 n_conv_channel_4: int):
        super().__init__()
        self.dec = conv_block(n_conv_channel_4, n_conv_channel_3)
        self.dec += conv_block(n_conv_channel_3, n_conv_channel_2)
        self.dec += conv_block(n_conv_channel_2, n_conv_channel_1)
        self.dec += conv_block(n_conv_channel_1, n_channel_out)

    def forward(self, x: torch.Tensor, seqlen: int) \
            -> Tuple[torch.Tensor, int]:
        # Input size: batch, channels
        x = x.repeat(seqlen, 1, 1)  # Size: length, batch, channels
        x = x.permute(1, 2, 0)  # Size: batch, channels, length
        x = self.dec(x)
        x = x.permute(0, 2, 1)  # Size: batch, length, channels
        return x


class ConvAutoencoder(Autoencoder):
    """Autoencoder with 1D-convolution layers."""

    def __init__(self, n_channel: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 n_conv_channel_4: int):
        super().__init__()
        self.encoder = ConvEncoder(n_channel, n_conv_channel_1,
                                   n_conv_channel_2, n_conv_channel_3,
                                   n_conv_channel_4)
        self.decoder = ConvDecoder(n_channel, n_conv_channel_1,
                                   n_conv_channel_2, n_conv_channel_3,
                                   n_conv_channel_4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.size(1)
        h = self.encoder(x)
        x = self.decoder(h, seqlen)
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


######################################
# Convolutional recurrent neural net #
######################################


class ConvRecEncoder(nn.Module):
    """Encoder with 1D-convolution and LTSM layers."""

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.conv_enc = conv_block(n_channel_in, n_conv_channel_1)
        self.conv_enc.append(nn.MaxPool1d(2))
        self.conv_enc += conv_block(n_conv_channel_1, n_conv_channel_2)
        self.conv_enc.append(nn.MaxPool1d(2))
        self.conv_enc += conv_block(n_conv_channel_2, n_conv_channel_3)
        self.conv_enc.append(nn.MaxPool1d(2))
        self.lstm_enc = nn.LSTM(n_conv_channel_3, lstm_hidden_size,
                                num_layers=n_lstm_layer, batch_first=True)

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input size: batch, length, channels
        x = x.permute(0, 2, 1)  # Size: batch, channels, length
        x = self.conv_enc(x)
        x = x.permute(0, 2, 1)  # Size: batch, length, channels
        x, (h, c) = self.lstm_enc(x)
        return x, (h, c)


class ConvRecDecoder(nn.Module):
    """Decoder with LTSM and 1D-convolution layers.

    Linear layer after LTSM to resize features to adjust channel dimension
    for convolution.
    """

    def __init__(self, n_channel_out: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.lstm_dec = nn.LSTM(lstm_hidden_size, lstm_hidden_size,
                                num_layers=n_lstm_layer, batch_first=True)
        self.linear_dec = nn.Linear(lstm_hidden_size, n_conv_channel_3)
        self.conv_dec = nn.Sequential(nn.Upsample(scale_factor=2))
        self.conv_dec += conv_block(n_conv_channel_3, n_conv_channel_2)
        self.conv_dec.append(nn.Upsample(scale_factor=2))
        self.conv_dec += conv_block(n_conv_channel_2, n_conv_channel_1)
        self.conv_dec.append(nn.Upsample(scale_factor=2))
        self.conv_dec += conv_block(n_conv_channel_1, n_channel_out)

    def forward(self, x: torch.Tensor, s: Tuple[torch.Tensor, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # Input size: batch, length, features
        x, _ = self.lstm_dec(x, s)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)  # Size: batch, features, length
        x = self.conv_dec(x)
        x = x.permute(0, 2, 1)  # Size: batch, length, channels
        return x


class ConvRecAutoencoder(Autoencoder):
    """Autoencoder with outer LTSM and inner 1D-convolution layers."""

    def __init__(self, n_channel_in: int, n_conv_channel_1: int,
                 n_conv_channel_2: int, n_conv_channel_3: int,
                 lstm_hidden_size: int, n_lstm_layer: int):
        super().__init__()
        self.encoder = ConvRecEncoder(n_channel_in, n_conv_channel_1,
                                      n_conv_channel_2, n_conv_channel_3,
                                      lstm_hidden_size, n_lstm_layer)
        self.decoder = ConvRecDecoder(n_channel_in, n_conv_channel_1,
                                      n_conv_channel_2, n_conv_channel_3,
                                      lstm_hidden_size, n_lstm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, c) = self.encoder(x)
        x = self.decoder(x, (h, c))
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        h = h.permute((1, 0, 2))  # Size: batch, LTSM layers, features
        return h


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_l1(seq_in: torch.Tensor, seq_out: torch.Tensor,
                lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss between two padded, masked sequences.

    The mask is used to remove the effect of padded values.
    """
    max_len = seq_out.size(1)
    batch_size = seq_in.size(0)
    # Output sequence length might be slightly different than input sequence
    # length
    seq_in = seq_in[:, :max_len, :]
    mask = mask[:, :max_len, :]
    # Sum of feature dimension
    err = torch.sum(torch.abs(seq_in - seq_out) * mask, 2)
    # Lengths
    lens = lengths.repeat((max_len, 1)).T
    return torch.sum(err / lens / batch_size)
