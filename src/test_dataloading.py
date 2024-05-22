"""Unit tests for the dataloading script."""


import random

from typing import List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import dataloading


@pytest.fixture
def data() -> List[Tuple[torch.Tensor, int]]:
    """Make a data input for the collate function."""
    x1 = torch.rand((1000, 10))
    x2 = torch.rand((2000, 10))
    x3 = torch.rand((3000, 10))
    return [(x1, 1000), (x2, 2000), (x3, 3000)]


@pytest.fixture
def dataset() -> Dataset:
    """Make a dataset from the SeismicSignals class."""
    return dataloading.SeismicSignals('data')


@pytest.fixture
def dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=128, shuffle=False,
                      collate_fn=dataloading.collate_fn)


def test_collate_fn(data: List[Tuple[torch.Tensor, int]]):
    """Test if the number of unmasked elements is correct."""
    _, lengths, mask = dataloading.collate_fn(data)
    assert mask.sum() == sum(le * 10 for le in lengths)


def test_dataset(dataset: Dataset):
    """Test if dataset items have the correct size."""
    X, cnt = None, 0
    while X is None and cnt < 100:
        i = random.randrange(len(dataset))
        X, _ = dataset[i]
        cnt += 1
    assert X.size()[1] == 3


def test_dataloader(dataloader: DataLoader):
    """Test if loop over dataloader runs till the end."""
    for i, _ in enumerate(dataloader):
        continue
    assert i == len(dataloader) - 1
