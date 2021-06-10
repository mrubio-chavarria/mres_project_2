#!/venv/bin python

"""
DESCRIPTION:
The classes needed to read and load the data from the read
folders.
"""

# Libraries
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from resquiggle_utils import parse_resquiggle, window_resquiggle


# Classes
class reshape2Tensor(object):
    """
    DESCRIPTION:
    A transform object to set the sample signals in the proper
    format.
    """
    # Methods
    def __init__(self, shape):
        """
        DESCRIPTION:
        Class constructor.
        :param shape: [tuple] the dimensions in which the signals
        should be fitted.
        """
        super().__init__()
        self.shape = shape
    
    def __call__(self, sample):
        sample['signal'] = torch.from_numpy(
            sample['signal'].astype(np.dtype('d'))
        ).view(*self.shape)
        return sample


class ONTDataset(Dataset):
    # Methods
    def __init__(self, reads_folder, reference_file, window_size=300, transform=None):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] folder containing the reads from which
        the windows should be obtained.
        :param reference_file: [str] fasta file containing the reference
        sequence to compare.
        :param transform: [reshape2Tensor] a transformation to set
        the vector dimensionality.
        """
        # Save parameters
        super().__init__()
        self.transform = transform
        self.folder = reads_folder
        self.reference = reference_file
        self.window_size = window_size
        # Load windows
        windows = load_windows(self.folder, self.reference, self.window_size)
        # Apply transform if needed
        if transform:
            self.windows = [transform(window) for window in windows]
        else:
            self.windows = windows
    
    def __len__(self):
        """
        DESCRIPTION:
        The size of the dataset is the number of windows.
        """
        return len(self.windows)
    
    def __getitem__(self, index):
        """
        DESRIPTION:
        :param index: [int] window position in the array.
        """
        return self.windows[index]
    
    def __iter__(self):
        """
        DESCRIPTION:
        """
        return iter(self.windows)
    
    def __next__(self):
        """
        DESCRIPTION:
        """
        return next(iter(self))


# Functions
def collate_text2int_fn(batch):
    """
    DESCRIPTION:
    The function to format the batches.
    :param batch: [list] the windows obtained from the resquiggle.
    :return: [dict] a dict with the sequences and the signals 
    separated and in matrix form.
    """
    # Relationship to convert between letters and labels
    relationship = {
        # Base: label
        '': 0,
        'A': 1,
        'T': 2,
        'G': 3,
        'C': 4
    }
    # Prepare the labels in pytorch format for CTC loss
    targets_lengths = []
    targets = []
    signals = []
    sequences = []
    fragments = []
    for item in batch:
        sequences.append(item['sequence'])
        targets_lengths.append(len(item['sequence']))
        targets.append(
            torch.Tensor(text2int(relationship, item['sequence']))
        )
        signals.append(item['signal'])
        fragments.append(item['fragments'])
    # Return the results
    return {
        'sequences': sequences,
        'targets': torch.cat(targets),
        'targets_lengths': tuple(targets_lengths),
        # Float format is important to make the data compatible with 
        # the model parameters
        'signals': torch.stack(signals).float(),
        'fragments': fragments
    }


def load_windows(folder, reference_file, window_size=300):
    """
    DESCRIPTION:
    Function to load all the windows extracted from a set of reads
    in a specified folder, and return them in the form of a dataset.
    :param folder: [str] the folder containing the annotated reads.
    :param reference_file: [str] the fasta file containing te reference
    sequence.
    :param window_size: [int] size of the window to analyse.
    :return: [list] windows (dicts) obtained from the folder reads.
    """
    # Read all the files in the folder
    files = os.listdir(folder)
    folder_windows = []
    for file in files:
        route = folder + '/' + file
        # Read resquiggle information
        segs, genome_seq, norm_signal = parse_resquiggle(route, reference_file)
        # Window the resquiggle signal
        file_windows = window_resquiggle(segs, genome_seq, 
            norm_signal, window_size)
        folder_windows.extend(file_windows)
    return folder_windows


def text2int(relationship, sequence):
    """
    DESCRIPTION:
    A function to translate a sequence of letters into a sequence of 
    numbers.
    :param relationship: [dict] the relationship between letters and 
    numbers. The numbers are ints. (key, value) -> (letter, number)
    :param sequence: [str] the sequence to convert.
    :return: [str] the sequence of numbers.
    """
    sequence = list(map(lambda x: relationship[x], sequence))
    return sequence