"""Unit tests for the modeling script"""


from typing import Tuple

import pytest
import torch

from signal_autoencoder.dataloading import collate_fn
from signal_autoencoder.modeling import (SignalAutoencoder,
				                         SignalEncoder,
					                     SignalDecoder,
                                         sequence_l1)


@pytest.fixture
def params() -> Tuple[int, Tuple[int, int, int, int, int]]:
    """Make model hyperparameters."""
    n_channel = 2
    n_conv_channel_1 = 64
    n_conv_channel_2 = 128
    n_conv_channel_3 = 256
    lstm_hidden_size = 512
    n_lstm_layer = 2
    return n_channel, (n_conv_channel_1, n_conv_channel_2, n_conv_channel_3,
                       lstm_hidden_size, n_lstm_layer)


@pytest.fixture
def batch(params: Tuple[int, Tuple[int, int, int, int, int]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a data batch."""
    n_channel, _ = params
    x1 = torch.rand((1000, n_channel))
    x2 = torch.rand((2000, n_channel))
    x3 = torch.rand((3000, n_channel))
    data = [(x1, 1000), (x2, 2000), (x3, 3000)]
    return collate_fn(data)


@pytest.fixture
def encoder(params: Tuple[int, Tuple[int, int, int, int, int]]) \
        -> SignalEncoder:
    """Make encoder model."""
    n_channel, other_params = params
    enc = SignalEncoder(n_channel, *other_params)
    return enc


@pytest.fixture
def decoder(params: Tuple[int, Tuple[int, int, int, int, int]]) \
        -> SignalDecoder:
    """Make decoder model."""
    n_channel, other_params = params
    enc = SignalDecoder(n_channel, *other_params)
    return enc


@pytest.fixture
def autoencoder(params: Tuple[int, Tuple[int, int, int, int, int]]) \
        -> SignalAutoencoder:
    """Make autoencoder model."""
    n_channel, other_params = params
    autoenc = SignalAutoencoder(n_channel, *other_params)
    return autoenc


def test_encoder(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 encoder: SignalEncoder):
    """Test if encoder output has the correct size."""
    x_in, _, _ = batch
    x, s = encoder(x_in)
    h, _ = s
    assert x.size() == torch.Size([x_in.size(0), x_in.size(1), h.size(2)])


def test_decoder(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 encoder: SignalEncoder,
                 decoder: SignalDecoder):
    """Test if decoder output has the correct size."""
    x_in, _, _ = batch
    x, s = encoder(x_in)
    x_out = decoder(x, s)
    assert x_in.size() == x_out.size()


def test_autoencoder(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     autoencoder: SignalAutoencoder):
    """Test if autoencoder output has the correct size."""
    x_in, _, _ = batch
    x_out = autoencoder(x_in)
    assert x_in.size() == x_out.size()


def test_sequence_l1(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     autoencoder: SignalAutoencoder):
    """Test if L1 loss between 2 batches of sequences runs smoothly."""
    x_in, lengths, mask = batch
    x_out = autoencoder(x_in)
    lossval = sequence_l1(x_in, x_out, lengths, mask)
    assert lossval
