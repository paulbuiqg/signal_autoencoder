"""Autoencoder for 1D multichannel signals of variable-length."""


from typing import Tuple

import torch
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, c) = self.encoder(x)
        x = self.decoder(x, (h, c))
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input, return the codes."""
        _, (h, _) = self.encoder(x)
        h = h.permute((1, 0, 2))  # Batch dimension 1st
        return torch.flatten(h, start_dim=1)


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
