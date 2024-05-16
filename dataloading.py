"""Data handling for 1D multichannel signals of variable-length."""


from typing import Tuple

import torch


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
        signals, batch_first=True, padding_value=-999)
    mask = torch.zeros(padded_sequences.size(), dtype=int)
    # Mask for reconstruction loss
    for i, le in enumerate(lengths):
        mask[i, :le, :] = 1
    return padded_sequences, lengths, mask
