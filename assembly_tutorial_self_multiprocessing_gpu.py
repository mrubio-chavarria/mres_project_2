#!/home/mario/anaconda3/envs/project2_venv/bin python

# Libraries
import multiprocessing
import os
import sys
import torch
from torch import cuda
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# Classes
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


class ResidualBlock(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout, n_features):
        """
        """
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(n_features)
        self.layer_norm2 = nn.LayerNorm(n_features)

    def forward(self, x):
        """
        :param x: [torch.Tensor] dimensionality: [batch_size, channels, mels, time]
        :return: [torch.Tensor] dimensionality: [batch_size, channels, mels, time]
        """
        residual = x
        output = self.layer_norm1(x.transpose(2, 3).contiguous())
        output = F.gelu(output.transpose(2, 3).contiguous())
        output = self.dropout1(output)
        output = self.cnn1(output)
        output = self.layer_norm2(output.transpose(2, 3).contiguous())
        output = F.gelu(output.transpose(2, 3).contiguous())
        output = self.dropout2(output)
        output = self.cnn2(output)        
        output += residual
        return output


class CNN_module(nn.Module):

    def __init__(self, n_layers, in_channels, out_channels, kernel_size, stride, dropout, n_features):
        """
        """
        super().__init__()
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(
                    ResidualBlock(in_channels, out_channels, kernel_size, stride, dropout, n_features)
                )
            else:
                self.layers.append(
                    ResidualBlock(out_channels, out_channels, kernel_size, stride, dropout, n_features)
                )
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x):
        output = self.model(x)
        return output
        

class RNN_module(nn.Module):
    """
    DESCRIPTION:
    """
    def __init__(self, input_size, hidden_size, n_layers, dropout, bidirectional=True):
        """"""
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.model = nn.GRU(input_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.layer_norm(x)
        output = F.gelu(output)
        output, _ = self.model(output)
        output = self.dropout(output)
        return output


class Decoder(nn.Module):
    """"""
    def __init__(self, input_size, n_classes, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_size, n_classes)
        )
    
    def forward(self, x):
        output = self.model(x)
        return output


class Network(nn.Module):
    """
    """
    def __init__(self, in_channels, kernel_size, n_conv_layers, n_rnn_layers, rnn_dim, n_kernels, n_features, n_classes, dropout):
        """"""
        super().__init__()
        n_features = n_features // 2  # Reduce the features because stride==2 in the first convolution
        self.initial_cnn = nn.Conv2d(in_channels, n_kernels, kernel_size, 2, padding=(3 - 1) // 2)
        self.cnn_module = CNN_module(n_conv_layers, n_kernels, n_kernels, kernel_size, 1, dropout, n_features)
        total_features = int(n_features * n_kernels)
        self.fully_connected = nn.Linear(total_features, rnn_dim)
        self.rnn_module = RNN_module(rnn_dim, 2 * rnn_dim, n_rnn_layers, dropout, True)
        self.decoder = Decoder(2 * rnn_dim, n_classes, dropout)

    def forward(self, x):
        output = self.initial_cnn(x)
        output = self.cnn_module(output)
        sizes = output.shape
        output = output.view(sizes[0], sizes[3], sizes[1] * sizes[2])
        output = self.fully_connected(output)
        output = self.rnn_module(output)
        output = self.decoder(output)
        return output



def limit_dataset(dataloader, limit):
    i = 0
    for item in dataloader:
        if i > limit:
            break
        yield item
        i += 1


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def data_processing(data, transforms):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        spec = transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes, targets


def train(model, train_data, test_data, parameters, device, sampler=None):
    """"""
    print(f'Training in GPU launched')
    # Create training parameters
    optimiser = optim.SGD(model.parameters(), parameters['learning_rate'])
    criterion = nn.CTCLoss(blank=28)
    scheduler = optim.lr_scheduler.OneCycleLR(optimiser, max_lr=parameters['learning_rate'], 
                                            steps_per_epoch=int(len(train_data)),
                                            epochs=parameters['n_epochs'],
                                            anneal_strategy='linear')
    # Launch training
    for epoch in range(parameters['n_epochs']):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, _data in enumerate(train_data):
            model.zero_grad()
            # Load data
            spectrograms, labels, input_lengths, label_lengths = _data
            # Compute model output
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            # Correct for DataParallel output
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)
            # Compute loss
            loss = criterion(output, labels, input_lengths, label_lengths)
            # Backpropagation
            loss.backward()
            optimiser.step()
            scheduler.step()
            # Print progress
            print('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch+1, batch_idx+1, len(train_data),
                100. * (batch_idx+1) / len(train_data), loss.item()))
        print('Epoch completed. Evaluating result...')
        test(model, test_data, criterion, device)


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def limit_dataset(dataloader, limit):
    i = 0
    for item in dataloader:
        if i > limit:
            break
        yield item
        i += 1


if __name__ == '__main__':

    if sys.argv[1] is None:
        raise ValueError('No CUDA found')

    os.environ['CUDA_VISIBLE_DEVICES']  = sys.argv[1]

    # Transformations to use in the data
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
    text_transform = TextTransform()

    # Parameters for training
    parameters = {
        "in_channels": 1,
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512, 
        "n_classes": 29,
        "n_features": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5E-4,
        "batch_size": 20,
        "n_epochs": 4,
        "kernel_size": 3,
        "n_kernels": 32
    }

    # Multiprocessing settings
    in_hpc = True
    

    # Import datasets
    if in_hpc:
        # When HPC
        train_dataset = torchaudio.datasets.LIBRISPEECH("/rds/general/user/mr820/home/project_2/librispeech_data", url="train-clean-100", download=True)
        test_dataset = torchaudio.datasets.LIBRISPEECH("/rds/general/user/mr820/home/project_2/librispeech_data", url="test-clean", download=True)
    else:
        # When local
        train_dataset = torchaudio.datasets.LIBRISPEECH("/home/mario/Projects/project_2/librispeech_data", url="train-clean-100", download=True)
        test_dataset = torchaudio.datasets.LIBRISPEECH("/home/mario/Projects/project_2/librispeech_data", url="test-clean", download=True)

    device = torch.device('cuda')
    print(f'Model training in {len(sys.argv[1].split(","))} GPUs' )

    # Create model
    model = Network(parameters['in_channels'],
                    parameters['kernel_size'],
                    parameters['n_cnn_layers'],
                    parameters['n_rnn_layers'],
                    parameters['rnn_dim'],
                    parameters['n_kernels'],
                    parameters['n_features'],
                    parameters['n_classes'],
                    parameters['dropout'])
    model = nn.DataParallel(model)
    model.to(device)    

    print('Model: ')
    print(model)
    # Training
    # Execute training
    print('Launching training')
    train_data = data.DataLoader(dataset=train_dataset,
                            shuffle=True,
                            batch_size=parameters['batch_size'],
                            collate_fn=lambda x: data_processing(x, train_audio_transforms))
    # Load test data
    test_data = data.DataLoader(dataset=test_dataset,
                            shuffle=True,
                            batch_size=parameters['batch_size'],
                            collate_fn=lambda x: data_processing(x, valid_audio_transforms))

    train_data = list(limit_dataset(train_data, 100))

    train(model, train_data, test_data, parameters, device)

    # # Save the model
    # if in_hpc:
    #     # When HPC
    #     path = "/rds/general/user/mr820/home/project_2/saved_models/model.pickle"
    #     torch.save(model.module.state_dict(), path)
    # else:
    #     # When local
    #     path = "/home/mario/Projects/project_2/saved_models/model.pickle"
    

