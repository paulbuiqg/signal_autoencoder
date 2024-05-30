"""Data handling for 1D multichannel signals of variable-length."""


import os
import pickle
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PADDING_VALUE = 2**31


def collate_fn(data: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to process and pad a batch of variable-length sequences.

    Args:
        data: List of tuples where each tuple contains a tensor representing
        a sequence and an integer representing its length.

    Returns:
        - padded_sequences: Tensor of shape (batch_size, max_seq_len, *)
          containing the padded sequences, where max_seq_len is the length of
          the longest sequence.
        - lengths: Tensor of shape (batch_size,) containing the lengths of
          each sequence in the batch.
        - mask: Binary tensor of shape (batch_size, max_seq_len, *) indicating
          the valid positions in the padded sequences (1 for valid, 0 for
          padding).
    """
    signals, lengths = zip(*data)
    lengths = torch.tensor(lengths)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        signals, batch_first=True, padding_value=PADDING_VALUE)
    mask = torch.zeros(padded_sequences.size(), dtype=int)
    # Mask for reconstruction loss
    for i, le in enumerate(lengths):
        mask[i, :le, :] = 1
    return padded_sequences, lengths, mask


class SeismicSignals(Dataset):
    """Dataset class for 3-channel seismic signals of any length."""

    def __init__(self, path: str, events: pd.DataFrame):
        super().__init__()
        self.path = path
        self.files, all_files = [], os.listdir(self.path)
        for event_id in events['eventID']:
            self.files += [f for f in all_files if str(event_id) in f]
        self.n_channel = 3
        self.mean = None
        self.std = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Union[None, Tuple[torch.Tensor, int]]:
        filepath = f'{self.path}/{self.files[i]}'
        try:
            with open(filepath, 'rb') as f:
                tr = pickle.load(f)
            signal_0, signal_1, signal_2 = tr[0].data, tr[1].data, tr[2].data
            # Differentiation to remove trend and center
            signal_0, signal_1, signal_2 = \
                np.diff(signal_0), np.diff(signal_1), np.diff(signal_2)
            # Truncate channel signals to same length
            seqlen = min(len(signal_0), len(signal_1), len(signal_2))
            signal = np.vstack((signal_0[:seqlen], signal_1[:seqlen],
                                signal_2[:seqlen]))
            # Length axis first for padding
            signal = torch.FloatTensor(signal.T)  # Size: length, channels
            # Normalization
            if self.mean is not None and self.std is not None:
                signal = (signal - self.mean) / self.std
            return signal, seqlen
        except BaseException as e:
            print(filepath, str(e))
            return None

    def plot(self, i: int):
        x, _ = self[i]
        fig, axs = plt.subplots(3)
        axs[0].plot(x[:, 0])
        axs[1].plot(x[:, 1])
        axs[2].plot(x[:, 2])
        fig.show()


def compute_signal_mean_and_std(device: str, dataloader: DataLoader) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer mean and standard deviation by channel."""
    progress_bar = tqdm(dataloader,
                        desc='Computing signal mean & std', unit='batch')
    means, vars = [], []
    for X, le, ma in progress_bar:
        X, le, ma = X.to(device), le.to(device), ma.to(device)
        # Size of X: batch, length, channels
        # Mean over time
        lens = le.repeat(3, 1).T  # Size: batch, channels
        m_ = (X * ma).sum(1) / lens  # Size: batch, channels
        m2_ = (X**2 * ma).sum(1) / lens  # Size: batch, channels
        # Mean over batch
        m = m_.mean(0)  # Size: channels
        s2 = (m2_ - m_**2).mean(0)  # Size: channels
        means.append(m)
        vars.append(s2)
    means = torch.vstack(means)
    vars = torch.vstack(vars)
    return means.median(0).values, torch.sqrt(vars.median(0).values)
