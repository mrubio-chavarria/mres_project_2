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
import torchaudio
from torch import nn


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


class PreONTDataset(Dataset):
    """
    DESCRIPTION:
    Dataset class to read and preprocess the raw ONT signal 
    from fast5 files.
    """
    def __init__(self, reads_folder, reference_file, window_size, max_number_windows=None):
        super().__init__()
        # Parameters
        # Mel Spectrogram
        sample_rate = 4000
        n_fft = 50
        window_length = n_fft
        hop_length = 1
        # Transforms to be applied over the whole signal
        preprocess_transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, 
                win_length=window_length, hop_length=hop_length)
        )
        self.windows = load_windows_with_features(reads_folder, reference_file, window_size, preprocess_transform)
        # Reduce the dataset if needed
        if max_number_windows is not None:
            self.windows = self.windows[:max_number_windows]

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

class RawONTDataset(Dataset):
    """
    DESCRIPTION:
    Dataset class to read and window the raw ONT signal 
    from fast5 files.
    """
    # Methods
    def __init__(self, reads_folder, reference_file, window_size=300, transform=None, max_number_windows=None):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] folder containing the reads from which
        the windows should be obtained.
        :param reference_file: [str] fasta file containing the reference
        sequence to compare.
        :param transform: [reshape2Tensor] a transformation to set
        the vector dimensionality.
        :param max_number_windows: [int] value to artificially reduce the
        dataset size.
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
        if transform is not None:
            self.windows = [transform(window) for window in windows]
        else:
            self.windows = windows
        # Reduce the dataset if needed
        if max_number_windows is not None:
            self.windows = self.windows[:max_number_windows]
    
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


class Dataset_3xr6(Dataset):
    """
    DESCRIPTION:
    Dataset to load and prepare the data in the 3xr6 dataset. 
    """
    # Methods
    def __init__(self, reads_folder='reads', reference_file='reference.fasta', window_size=300, max_number_windows=None, flowcell=None):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] route to the folder containing the flowcell folders.
        :param reference_file: [str] the file containing the reference sequence in fasta
        format.
        :param window_size: [int] size in which the reads should sliced. 
        :param max_number_wndows: [int] parameter to artificially decrease the size of the 
        dataset.
        :param flowcell: [str] a param to specify if only one flowcell should be analysed.        
        """
        # Helper function
        def file_hq_filter(file):
            """
            DESCRIPTION:
            A helper function to obtain only those reads with a high-quality alignment.
            :param file: [str] read filename.
            :return: [bool] True if the read is of high quality.
            """
            return file.startswith('Q20') and file.endswith('fast5')

        # Save parameters
        super().__init__()
        self.reads_folder = reads_folder
        self.reference = reference_file
        self.window_size = window_size
        self.max_number_windows = max_number_windows
        # Obtain the high_quality files with the reads
        self.read_files = []
        if flowcell is None:
            for flowcell in os.listdir(self.reads_folder):
                flowcell_file = reads_folder + '/' + flowcell + '/' + 'single'
                folders = [folder for folder in os.listdir(flowcell_file) 
                    if not (folder.endswith('txt') or folder.endswith('index'))]
                for folder in folders:
                    folder_file = flowcell_file + '/' + folder
                    files = filter(lambda file: file_hq_filter(file), os.listdir(folder_file))
                    files = map(lambda file: folder_file + '/' + file, files)
                    self.read_files.extend(files)
        else:
            flowcell_file = reads_folder + '/' + flowcell + '/' + 'single'
            folders = [folder for folder in os.listdir(flowcell_file) 
                if not (folder.endswith('txt') or folder.endswith('index'))]
            for folder in folders:
                folder_file = flowcell_file + '/' + folder
                files = filter(lambda file: file_hq_filter(file), os.listdir(folder_file))
                files = map(lambda file: folder_file + '/' + file, files)
                self.read_files.extend(files)
        # Load windows
        self.windows = load_windows(self.read_files, self.reference, self.window_size)
        # Reduce the dataset if needed
        if max_number_windows is not None:
            self.windows = self.windows[:max_number_windows]
        # Add 1 dimensiones because there is one channel
        [window.update({'signal': torch.unsqueeze(window['signal'], dim=0)}) for window in self.windows]

    
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


class Dataset_3xr6_transformed(Dataset):
    """
    DESCRIPTION:
    Version of Dataset_3xr6 with the ability to execute transformation  at the read
    level.
    """
    # Methods
    def __init__(self, reads_folder='reads', reference_file='reference.fasta', window_size=300, max_number_windows=None, transform=None):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] route to the folder containing the flowcell folders.
        :param reference_file: [str] the file containing the reference sequence in fasta
        format.
        :param window_size: [int] size in which the reads should sliced. 
        :param max_number_wndows: [int] parameter to artificially decrease the size of 
        the dataset.
        :param transform: [nn.Sequential] the transformations to apply at read 
        level.        
        """
        # Helper function
        def file_hq_filter(file):
            """
            DESCRIPTION:
            A helper function to obtain only those reads with a high-quality alignment.
            :param file: [str] read filename.
            :return: [bool] True if the read is of high quality.
            """
            return file.startswith('kHQk') and file.endswith('fast5')

        # Save parameters
        super().__init__()
        self.reads_folder = reads_folder
        self.reference = reference_file
        self.window_size = window_size
        self.max_number_windows = max_number_windows
        self.transform = transform
        # Obtain the high_quality files with the reads
        self.read_files = []
        for flowcell in os.listdir(self.reads_folder):
            flowcell_file = reads_folder + '/' + flowcell + '/' + 'single'
            folders = [folder for folder in os.listdir(flowcell_file) 
                if not (folder.endswith('txt') or folder.endswith('index'))]
            for folder in folders:
                folder_file = flowcell_file + '/' + folder
                files = filter(lambda file: file_hq_filter(file), os.listdir(folder_file))
                files = map(lambda file: folder_file + '/' + file, files)
                self.read_files.extend(files)
        # Load windows
        self.windows = load_windows_with_transform(self.read_files, self.reference, self.window_size, self.transform)
        # Reduce the dataset if needed
        if max_number_windows is not None:
            self.windows = self.windows[:max_number_windows]
        # Add 1 dimensiones because there is one channel
        [window.update({'signal': torch.unsqueeze(window['signal'], dim=0)}) for window in self.windows]
    
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
        '$': 0,
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
        'targets': torch.cat(targets).type(torch.int8),
        'targets_lengths': tuple(targets_lengths),
        # Float format is important to make the data compatible with 
        # the model parameters
        'signals': torch.stack(signals).float(),
        'fragments': fragments
    }


def load_windows(read_files, reference_file, window_size=300):
    """
    DESCRIPTION:
    Function to load all the windows extracted from a set of reads
    in a specified folder, and return them in the form of a dataset.
    :param read_files: [list] files with the reads to window. Every file
    shoudl contain the complete path.
    :param reference_file: [str] the fasta file containing te reference
    sequence.
    :param window_size: [int] size of the window to analyse.
    :return: [list] windows (dicts) obtained from the folder reads.
    """
    # Read all the files in the folder
    total_windows = []
    for route in read_files:
        # Read resquiggle information
        segs, genome_seq, norm_signal = parse_resquiggle(route, reference_file)
        # Window the resquiggle signal
        file_windows = window_resquiggle(segs, genome_seq, norm_signal, window_size)
        total_windows.extend(file_windows)
    return total_windows


def load_windows_with_transform(read_files, reference_file, window_size=300, transform=None):
    """
    DESCRIPTION:
    Version of the function above but with the option to apply a 
    transformation at read level.
    :param reads_folder: [str] the folder containing the annotated reads.
    :param reference_file: [str] the fasta file containing te reference
    sequence.
    :param window_size: [int] size of the window to analyse.
    :param transform: [nn.Sequential] transforms to apply 
    over the whole signal before windowing.
    :return: [list] windows (dicts) obtained from the folder reads.
    """
    # Read all the files in the folder
    total_windows = []
    for route in read_files:
        # Read resquiggle information
        segs, genome_seq, norm_signal = parse_resquiggle(route, reference_file)
        # Window the resquiggle signal
        file_windows = window_resquiggle(segs, genome_seq, 
            norm_signal, window_size)
        # Apply preprocessing over the normalised signal
        transformed_norm_signal = transform(torch.FloatTensor(norm_signal))[:, :-1]
        # This function transforms directly to tensor the original signal
        for window in file_windows:
            # Window the transformed signal, save in the position of the original signal
            window.update({'signal': transformed_norm_signal[:, window['signal_indeces'][0]:window['signal_indeces'][1]]})
            # window['signal'] = torch.from_numpy(window['signal'].astype(np.dtype('d'))).view(-1, 1)
            # Delete non-necessary fields
            del window['signal_indeces']
        total_windows.extend(file_windows)
    return total_windows


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