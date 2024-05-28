"""Unit tests for the modeling script"""


from typing import Tuple

import pytest
import torch

from signal_autoencoder.dataloading import collate_fn
from signal_autoencoder.modeling import (ConvAutoencoder,
                                         ConvDecoder,
                                         ConvEncoder,
                                         ConvRecAutoencoder,
                                         ConvRecEncoder,
                                         ConvRecDecoder,
                                         sequence_l1)


############
# Fixtures #
############


@pytest.fixture
def conv_rec_params() -> Tuple[int, Tuple[int, int, int, int, int]]:
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
def conv_params() -> Tuple[int, Tuple[int, int, int, int, int]]:
    """Make model hyperparameters."""
    n_channel = 2
    n_conv_channel_1 = 64
    n_conv_channel_2 = 128
    n_conv_channel_3 = 256
    n_conv_channel_4 = 512
    return n_channel, (n_conv_channel_1, n_conv_channel_2, n_conv_channel_3,
                       n_conv_channel_4)


@pytest.fixture
def batch(conv_rec_params: Tuple[int, Tuple[int, int, int, int, int]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a data batch."""
    n_channel, _ = conv_rec_params
    x1 = torch.rand((1000, n_channel))
    x2 = torch.rand((2000, n_channel))
    x3 = torch.rand((3000, n_channel))
    data = [(x1, 1000), (x2, 2000), (x3, 3000)]
    return collate_fn(data)


@pytest.fixture
def conv_encoder(conv_params: Tuple[int, Tuple[int, int, int, int]]) \
        -> ConvEncoder:
    """Make convolutional encoder model."""
    n_channel, other_params = conv_params
    enc = ConvEncoder(n_channel, *other_params)
    return enc


@pytest.fixture
def conv_decoder(conv_params: Tuple[int, Tuple[int, int, int, int]]) \
        -> ConvDecoder:
    """Make convolutional decoder model."""
    n_channel, other_params = conv_params
    dec = ConvDecoder(n_channel, *other_params)
    return dec


@pytest.fixture
def conv_autoencoder(conv_params: Tuple[int, Tuple[int, int, int, int]]) \
        -> ConvAutoencoder:
    """Make convolutional autoencoder model."""
    n_channel, other_params = conv_params
    autoenc = ConvAutoencoder(n_channel, *other_params)
    return autoenc


@pytest.fixture
def conv_rec_encoder(
    conv_rec_params: Tuple[int, Tuple[int, int, int, int, int]]
) -> ConvRecEncoder:
    """Make convolutional recurrent encoder model."""
    n_channel, other_params = conv_rec_params
    enc = ConvRecEncoder(n_channel, *other_params)
    return enc


@pytest.fixture
def conv_rec_decoder(
    conv_rec_params: Tuple[int, Tuple[int, int, int, int, int]]
) -> ConvRecDecoder:
    """Make convolutional recurrent decoder model."""
    n_channel, other_params = conv_rec_params
    enc = ConvRecDecoder(n_channel, *other_params)
    return enc


@pytest.fixture
def conv_rec_autoencoder(
    conv_rec_params: Tuple[int, Tuple[int, int, int, int, int]]
) -> ConvRecAutoencoder:
    """Make convolutional recurrent autoencoder model."""
    n_channel, other_params = conv_rec_params
    autoenc = ConvRecAutoencoder(n_channel, *other_params)
    return autoenc


############################
# Convolutional neural net #
############################


def test_conv_encoder(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      conv_params: Tuple[int, Tuple[int, int, int, int, int]],
                      conv_encoder: ConvEncoder):
    """Test if convolutional encoder output has the correct size."""
    _, (_, _, _, d) = conv_params
    x, _, _ = batch
    h = conv_encoder(x)
    assert h.size() == torch.Size([x.size(0), d])


def test_conv_decoder(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      conv_encoder: ConvDecoder,
                      conv_decoder: ConvDecoder):
    """Test if convolutional decoder output has the correct size."""
    x_in, _, _ = batch
    h = conv_encoder(x_in)
    x_out = conv_decoder(h, x_in.size(1))
    assert x_in.size() == x_out.size()


def test_conv_rec_autoencoder(
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        conv_autoencoder: ConvAutoencoder):
    """Test if convolutional autoencoder output has the correct size."""
    x_in, _, _ = batch
    x_out = conv_autoencoder(x_in)
    assert x_in.size() == x_out.size()


######################################
# Convolutional recurrent neural net #
######################################


def test_conv_rec_encoder(
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        conv_rec_encoder: ConvRecEncoder):
    """Test if convolutional recurrent encoder output has the correct size."""
    x_in, _, _ = batch
    x, s = conv_rec_encoder(x_in)
    h, _ = s
    assert x.size() == torch.Size([x_in.size(0), x_in.size(1), h.size(2)])


def test_conv_rec_decoder(
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        conv_rec_encoder: ConvRecEncoder,
        conv_rec_decoder: ConvRecDecoder):
    """Test if convolutional recurrent decoder output has the correct size."""
    x_in, _, _ = batch
    x, s = conv_rec_encoder(x_in)
    x_out = conv_rec_decoder(x, s)
    assert x_in.size() == x_out.size()


def test_conv_rec_autoencoder(
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        conv_rec_autoencoder: ConvRecAutoencoder):
    """Test if conv recurrent autoencoder output has the correct size."""
    x_in, _, _ = batch
    x_out = conv_rec_autoencoder(x_in)
    assert x_in.size() == x_out.size()


def test_sequence_l1(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     conv_autoencoder: ConvAutoencoder,
                     conv_rec_autoencoder: ConvRecAutoencoder):
    """Test if L1 loss between 2 batches of sequences runs smoothly."""
    x_in, lengths, mask = batch
    x_out = conv_rec_autoencoder(x_in)
    lossval = sequence_l1(x_in, x_out, lengths, mask)
    assert lossval
    x_out = conv_autoencoder(x_in)
    lossval = sequence_l1(x_in, x_out, lengths, mask)
    assert lossval
