#!/venv/bin python

"""
DESCRIPTION:
This file contains wrappers and variations on DataLoader.
"""

# Libraries
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from resquiggle_utils import parse_resquiggle, window_resquiggle
from torch import nn


class CombinedDataLoader:
    """
    DESCRIPTION:
    """
    # Methods
    def __init__(self, *args):
        """
        DESCRIPTION:
        """
        self.current_dataloader = 0
        self.dataloaders = args

    def __next__(self):
        """
        DESCRIPTION:
        """
        next_batch = next(iter(self.dataloaders[self.current_dataloader]))
        self.current_dataloader = (self.current_dataloader + 1) % len(self.dataloaders) 
        return next_batch


class CustomisedDataLoader:
    """
    DESCRIPTION:
    """
    # Methods
    def __init__(self, dataset, batch_size, sampler, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.collate_fn = collate_fn

    def __iter__(self):
        sampled_data = self.sampler(self.dataset, self.batch_size)
        for batch in sampled_data:
            if not batch:
                raise StopIteration
            yield self.collate_fn(batch)

    def __next__(self):
        return next(iter(self))

