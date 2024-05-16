"""Unit tests for the dataloading script."""


from typing import List, Tuple

import pytest
import torch

from dataloading import collate_fn


@pytest.fixture
def data() -> List[Tuple[torch.Tensor, int]]:
    """Make a data input for the collate function."""
    x1 = torch.rand((1000, 10))
    x2 = torch.rand((2000, 10))
    x3 = torch.rand((3000, 10))
    return [(x1, 1000), (x2, 2000), (x3, 3000)]


def test_collate_fn(data: List[Tuple[torch.Tensor, int]]):
    """Test if the number of unmasked elements is correct."""
    _, lengths, mask = collate_fn(data)
    assert mask.sum() == sum(le * 10 for le in lengths)
