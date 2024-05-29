"""Data handling for 1D multichannel signals of variable-length."""


import os
import pickle
from typing import Tuple, Union

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
            seqlen = min(len(signal_0), len(signal_1), len(signal_2))
            signal = np.vstack((signal_0[:seqlen], signal_1[:seqlen],
                                signal_2[:seqlen]))
            signal = torch.FloatTensor(signal.T)
            # Centering (offset)
            signal = signal - signal.mean(dim=0)
            # Normalization
            if self.mean is not None and self.std is not None:
                signal = (signal - self.mean) / self.std
            return signal, seqlen
        except BaseException as e:
            print(filepath, str(e))
            return None

    def set_mean_and_std(self, mean: float, std: float):
        """Set mean and std values."""
        self.mean = mean
        self.std = std


def compute_signal_mean_and_std(device: str, dataloader: DataLoader) \
        -> Tuple[float, float]:
    """Loop over dataset to infer mean and standard deviation."""
    progress_bar = tqdm(
        dataloader, desc='Computing signal mean & std', unit='batch')
    cum_m, cum_m2, batch_cnt = 0, 0, 0
    for X, _, ma in progress_bar:
        X, ma = X.to(device), ma.to(device)
        X = X.flatten()[ma.flatten() == 1]
        m = X.mean()
        m2 = (X**2).sum()
        cum_m = (cum_m * batch_cnt + m) / (batch_cnt + 1)
        cum_m2 = (cum_m2 * batch_cnt + m2) / (batch_cnt + 1)
        batch_cnt += 1
        progress_bar.set_postfix_str(f'mean={cum_m:>8f}')
    mean = cum_m
    std = torch.sqrt(cum_m2 - mean**2)
    return mean.item(), std.item()
