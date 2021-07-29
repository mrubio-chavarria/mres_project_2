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
from torch.utils import data
from torch.utils.data import Dataset
from resquiggle_utils import parse_resquiggle, window_resquiggle
from torch import nn
import random
from tqdm import tqdm


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


class CustomisedSampler(torch.utils.data.Sampler):
    """
    DESCRIPTION:
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        DESCRIPTION:
        Class constructor.
        :param dataset: [CombinedDataset] the dataset to create the batches. 
        :param batch_size: [int] the number of samples in the batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # Detect the number of datasets
        datasets = [attr for attr in dir(self.dataset) if attr.startswith('windows')]
        datasets = list(sorted(datasets, key=lambda x: int(x.split('_')[1])))
        # Create the list to iterate
        batches_by_dataset = []
        for dataset_name in datasets:
            # Get dataset
            dataset = getattr(self.dataset, dataset_name)
            # Shuffle dataset
            if self.shuffle:
                random.shuffle(dataset)
            # Compute the batches
            dataset_id = int(dataset_name.split('_')[1])
            n_batches = len(dataset) // self.batch_size
            dataset_batches = [(self.batch_size*i, self.batch_size*(i+1)) 
                for i in range(n_batches)]
            batches_by_dataset.append(dataset_batches)
        # Equal number of batches
        n_last = len(batches_by_dataset[-1])
        batches_by_dataset = [batches[:n_last] for batches in batches_by_dataset]
        for i in range(n_last * len(datasets)):
            dataset = datasets[i % len(datasets)]
            dataset = getattr(self.dataset, dataset)
            start, end = batches_by_dataset[i % len(datasets)].pop()
            item = dataset[start:end]
            yield item


class CombinedDataset(Dataset):
    """
    DESCRIPTION:
    A dataset to combine datasets of multiple window sizes.
    """
    # Methods
    def __init__(self, *args):
        """
        DESCRIPTION:
        Class constructor.
        :param args: [list] datasets to combine. They are to be of the structure
        shown with Dataset_ap or Dataset_3xr6.
        """
        super().__init__()
        self.datasets = args
        [setattr(self, f'windows_{dataset.window_size}', dataset.windows) 
            for dataset in self.datasets]

    def __len__(self):
        """
        DESCRIPTION:
        The size of the dataset is the number of windows in all datasets.
        """
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, args):
        """
        DESRIPTION:
        Method to support indexing in the dataset.
        :param window_size: [int] window_size that characterise the dataset.
        :param index: [int] window position in the array.
        """
        window_size, index = args
        return getattr(self, f'windows_{window_size}')[index]


class Dataset_ap(Dataset):
    """"
    DESCRIPTION:
    Dataset to load and prepare the data in the acinetobacter.
    pittii (ap) dataset
    """
    # Methods
    def __init__(self, reads_folder='reads', reference_file='reference.fasta', window_size=300, max_number_windows=None, flowcell=None, hq_value='Q20', max_reads=4000):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] route to the folder containing the flowcell folders.
        :param reference_file: [str] the file containing the reference sequence in fasta
        format.
        :param window_size: [int] size in which the reads should be sliced. 
        :param max_number_wndows: [int] parameter to artificially decrease the size of the 
        dataset.
        :param flowcell: [str] a param to specify if only one flowcell should be analysed.
        If None, all the flowcells are loaded. It should of the form 'flowcellX'.
        """
        # Helper function
        def file_hq_filter(file):
            """
            DESCRIPTION:
            A helper function to obtain only those reads with a high-quality alignment.
            :param file: [str] read filename.
            :return: [bool] True if the read is of high quality.
            """
            return file.startswith(hq_value) and file.endswith('fast5')
        
        super().__init__()
        self.reads_folder = reads_folder
        self.reference = reference_file
        self.window_size = window_size
        self.max_number_windows = max_number_windows
        # Obtain the high_quality files with the reads
        self.read_files = []
        if flowcell is None:
            for flowcell in os.listdir(self.reads_folder):
                flowcell_folder = reads_folder + '/' + flowcell + '/' + 'single'
                files = [file for file in os.listdir(flowcell_folder) 
                    if not (file.endswith('txt') or file.endswith('index'))]
                files = filter(lambda file: file_hq_filter(file), files)
                files = map(lambda file: flowcell_folder + '/' + file, files)
                self.read_files.extend(files)
        else:
            flowcell_folder = reads_folder + '/' + flowcell + '/' + 'single'
            files = [file for file in os.listdir(flowcell_folder) 
                if not (file.endswith('txt') or file.endswith('index'))]
            files = filter(lambda file: file_hq_filter(file), files)
            files = map(lambda file: flowcell_folder + '/' + file, files)
            self.read_files.extend(files)
        # Limit the number of reads
        if max_reads is not None:
            if len(self.read_files) >= max_reads:
                self.read_files = random.sample(self.read_files, max_reads)
        # Load windows
        self.windows = load_windows(self.read_files, self.reference, self.window_size, bandwidth=24000)
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


class Dataset_3xr6(Dataset):
    """
    DESCRIPTION:
    Dataset to load and prepare the data in the 3xr6 dataset. 
    """
    # Methods
    def __init__(self, reads_folder='reads', reference_file='reference.fasta', window_size=300, max_number_windows=None, flowcell=None, hq_value='Q20'):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] route to the folder containing the flowcell folders.
        :param reference_file: [str] the file containing the reference sequence in fasta
        format.
        :param window_size: [int] size in which the reads should be sliced. 
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
            return file.startswith(hq_value) and file.endswith('fast5')

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


class Dataset_3xr6_alternating(Dataset):
    """
    DESCRIPTION:
    This dataset is a copy of the previous one but with alternating 
    window sizes between batches.
    """
    # Methods
    def __init__(self, reads_folder='reads', reference_file='reference.fasta', window_sizes=[300], max_number_windows=None, flowcell=None, hq_value='Q20'):
        """
        DESCRIPTION:
        Class constructor.
        :param reads_folder: [str] route to the folder containing the flowcell folders.
        :param reference_file: [str] the file containing the reference sequence in fasta
        format.
        :param window_sizes: [list] sizes in which the reads should be sliced, a set of 
        windows will be created for every size. This is considered for batching.
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
            return file.startswith(hq_value) and file.endswith('fast5')

        # Save parameters
        super().__init__()
        self.reads_folder = reads_folder
        self.reference = reference_file
        self.window_sizes = window_sizes
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
        for window_size in self.window_sizes:
            setattr(self, f'windows_s{window_size}', load_windows(self.read_files, self.reference, window_size))
        for window_size in self.window_sizes:
            # Reduce the dataset if needed
            if max_number_windows is not None:
                setattr(self, f'windows_s{window_size}', getattr(self, f'windows_s{window_size}')[:max_number_windows])
            # Add 1 dimensiones because there is one channel
            [window.update({'signal': torch.unsqueeze(window['signal'], dim=0)}) for window in getattr(self, f'windows_s{window_size}')]

    def __len__(self):
        """
        DESCRIPTION:
        The size of the dataset is the number of windows.
        """
        return sum([len(getattr(self, f'windows_s{window_size}')) for window_size in self.window_sizes])
    
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
        'A': 0,
        'T': 1,
        'G': 2,
        'C': 3,
        '$': 4
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
            torch.Tensor(text2int(relationship, item['sequence'])).int()
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


def load_windows(read_files, reference_file, window_size=300, bandwidth=6000, read=True):
    """
    DESCRIPTION:
    Function to load all the windows extracted from a set of reads
    in a specified folder, and return them in the form of a dataset.
    :param read_files: [list] files with the reads to window. Every file
    shoudl contain the complete path.
    :param reference_file: [str] the fasta file containing te reference
    sequence.
    :param window_size: [int] size of the window to analyse.
    :param read: [bool] flag to indicate if the resquiggling should be 
    computed again or read from the files.
    :return: [list] windows (dicts) obtained from the folder reads.
    """
    # Read all the files in the folder
    total_windows = []
    print('-------------------------------------------------')
    print('Loading reads')
    print('-------------------------------------------------')
    skipped_reads = 0
    for route in tqdm(read_files):
        # Read resquiggle information
        try:
            segs, genome_seq, norm_signal = parse_resquiggle(route, reference_file, bandwidth, read)
        except:
            # In some reads the resquiggling was not successful
            skipped_reads += 1
            continue
        # Window the resquiggle signal
        file_windows = window_resquiggle(segs, genome_seq, norm_signal, window_size)
        total_windows.extend(file_windows)
    print('-------------------------------------------------')
    print('Skipped reads:', str(skipped_reads))
    print('-------------------------------------------------')
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