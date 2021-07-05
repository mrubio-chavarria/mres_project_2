#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The code below solves the problem of correcting the basecalled
sequence based on the semi-global alignment with the reference
sequence.
"""

# Libraries
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from datasets import ONTDataset, reshape2Tensor, collate_text2int_fn
from models import ResidualBlockIV, TCN_module, LSTM_module, ClassifierGELU, ResNet
from metrics import cer, _levenshtein_distance, char_errors
from ont_fast5_api.fast5_interface import get_fast5_file
import os
from torch import multiprocessing as mp
from bnlstm import LSTM


# Classes
class Network(nn.Module):
    """
    DESCRIPTION:
    Final model.
    """
    # Methods
    def __init__(self, TCN_parameters, LSTM_parameters, decoder_parameters):
        super().__init__()
        self.TCN_parameters = TCN_parameters
        self.LSTM_parameters = LSTM_parameters
        self.decoder_parameters = decoder_parameters
        self.convolutional_module = TCN_module(**self.TCN_parameters)
        self.recurrent_module = LSTM_module(**self.LSTM_parameters)
        self.decoder = ClassifierGELU(**self.decoder_parameters)

    def forward(self, input_sequence):
        """
        Forward pass.
        :param input_sequence: [torch.Tensor] batch to feed the 
        model. 
        Dimensions: [batch_size, input_dimensionality, sequence_length]
        """
        output = self.convolutional_module(input_sequence)
        output = self.recurrent_module(output)
        output = self.decoder(output)
        return output


# Functions
def length2indices(window):
    indeces = [0]
    for i in range(len(window)):
        indeces.append(window[i] + indeces[i])
    return indeces


def decoder(probabilities_matrix, segments_lengths):
    max_probabilities = torch.argmax(probabilities_matrix, dim=2)
    letters = ['$', 'A', 'T', 'G', 'C']
    windows = [length2indices(window) for window in segments_lengths]
    for i in range(len(probabilities_matrix)):
        # Output probabilities to sequence
        sequence = [[letters[prob] for prob in max_probabilities[i].tolist()][windows[i][index]:windows[i][index+1]]
            for index in range(len(windows[i])-1)]
        sequence = ''.join([''.join(list(set(segment))) for segment in sequence])
        yield sequence.replace('$', '')


def init_weights(module):
    """
    DESCRIPTION:
    Function to initialise weights with xavier uniform.
    :param module: [nn.Module] the module whose weights you are initialising.
    """
    if type(module) in [nn.Conv1d, nn.Linear]:
        nn.init.xavier_uniform_(module.weight)
    if type(module) is nn.LSTM:
        [nn.init.xavier_uniform_(getattr(module, attr)) for attr in dir(module) if attr.startswith('weight_')]


def launch_training(model, train_data, rank=0, sampler=None, **kwargs):
    """
    DESCRIPTION:
    UPDATE
    A function to launch the model training.
    :param rank: [int] index of the process executing the function.
    :param sampler: [DistributedSampler] sampler to control batch reordering
    after eery epoch in the case of multiprocessing.
    """
    # Create optimiser
    if kwargs.get('optimiser', 'SGD') == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-4),
                                    momentum=kwargs.get('momemtum', 0.99))
    elif kwargs.get('optimiser', 'SGD') == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-4),
                                    weight_decay=kwargs.get('weight_decay', 0.99))
    elif kwargs.get('optimiser') == 'RMSprop':
        optimiser = torch.optim.RMSprop(model.parameters(),
                                        lr=kwargs.get('learning_rate', 1E-4),
                                        weight_decay=kwargs.get('weight_decay', 1E-3),
                                        momentum=kwargs.get('momemtum'))
    else:
        raise ValueError('Invalid optimiser selected')
    # Create scheduler
    if kwargs.get('scheduler') is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
                                                        max_lr=kwargs.get('max_learning_rate', 1E-2),
                                                        steps_per_epoch=len(train_data),
                                                        epochs=kwargs.get('epochs', 30))
    # Prepare training
    sequence_length, batch_size = kwargs.get('sequence_length'), kwargs.get('batch_size')
    sequences_lengths = tuple([sequence_length] * batch_size)
    log_softmax = nn.LogSoftmax(dim=2)
    loss_function = nn.CTCLoss()
    initialisation_loss_function = nn.CrossEntropyLoss()
    initialisation_epochs = range(kwargs.get('n_initialisation_epochs', 1))
    # Train
    for epoch in range(kwargs.get('epochs', 30)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_id, batch in enumerate(train_data):
            # Clean gradient
            model.zero_grad()
            # Move data to device
            target_segments = batch['fragments']
            target_sequences = batch['sequences']
            targets_lengths = batch['targets_lengths']
            batch, target = batch['signals'], batch['targets']
            if batch.shape[0] != batch_size:
                continue
            # Forward pass
            output = model(batch)
            # Loss
            # Set different loss to initialise
            if epoch in initialisation_epochs:
                # Initialisation
                # Loss function: CrossEntropy
                # Create the labels
                fragments = target_segments
                new_targets = []
                total = 0
                new_targets = []
                for i in range(len(fragments)):
                    new_target = [it for sb in [[target[total+j].tolist()] * fragments[i][j] for j in range(len(fragments[i]))] for it in sb]
                    new_targets.append(new_target)
                    total += len(fragments[i])
                targets = torch.stack([torch.LongTensor(target) for target in new_targets])
                # Compute the loss
                loss = initialisation_loss_function(output.view(batch_size, -1, sequence_length), targets)
            else:
                # Regular
                # Loss function: CTC
                # Compute the loss
                loss = loss_function(
                    log_softmax(output.reshape(sequence_length, batch_size, -1)),
                    target,
                    sequences_lengths,
                    targets_lengths
                )
            # Backward pass
            loss.backward()
            # Gradient step
            optimiser.step()
            if kwargs.get('scheduler') is not None:
                scheduler.step()
            # Decode output
            output_sequences = list(decoder(output, target_segments))
            error_rates = [cer(target_sequences[i], output_sequences[i]) for i in range(len(output_sequences))]
            avg_error = sum(error_rates) / len(error_rates)
            # Show progress
            print('----------------------------------------------------------------------------------------------------------------------')
            print(f'First target: {target_sequences[0]}\nFirst output: {output_sequences[0]}')
            if kwargs.get('scheduler') is not None:
                print(f'Process: {rank} Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error} Learning rate: {optimiser.param_groups[0]["lr"]}')
            else:
                print(f'Process: {rank} Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error}')


def train(model, train_dataset, algorithm='single', n_processes=3, **kwargs):
    """
    DESCRIPTION:
    UPDATE
    A wrapper to control the use of multiprocessing Hogwild algorithm from single
    node traditional training.
    :param algorithm: [str] the algorithm to train. Classic single-node and CPU 
    trainin, or single-node multi CPU Hogwild.
    :param model: [torch.nn.Module] the model to train.
    """
    # Prepare the model
    model.train()
    print('Training started')
    # Select training algorithm
    if algorithm == 'single':
        # Prepare the data
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_text2int_fn)
        # Start training
        launch_training(model, train_data, **kwargs)
    elif algorithm == 'Hogwild':
        # Start training
        # We are not setting a blockade per epoch
        model.share_memory()
        processes = []
        for rank in range(n_processes):
            train_sampler = DistributedSampler(train_dataset, num_replicas=n_processes, rank=rank)
            train_data = DataLoader(dataset=train_dataset,
                            sampler=train_sampler,
                            batch_size=kwargs['batch_size'],
                            collate_fn=collate_text2int_fn)
            print(f'Process {rank} launched')
            process = mp.Process(target=launch_training, args=(model, train_data, rank, train_sampler), kwargs=kwargs)
            process.start()
            processes.append(process)
        for process in processes:
            process.join()


if __name__ == "__main__":
    
    """
    NOTES TO CONSIDER:
    - The data has not been rescaled, although Tombo normalised the signal.
    It is not the same.
    - Because we only work with CPUs, the device parameter is not defined. 
    The reason is that the default device for pytorch, according to the 
    version used (torch==1.9.0+cpu) is CPU.
    """

    # Set fast5 and reference
    # reads_folder = "databases/synthetic_flappie_r941_native_3xr6/reads"
    reference_file = "databases/natural_flappie_r941_native_ap_toy/reference.fasta"
    
    # Load the train and test datasets
    transform = reshape2Tensor((1, -1))
    batch_size = 10
    window_size = 311
    train_folder = "databases/natural_flappie_r941_native_ap_toy/train_reads"
    max_number_windows = 300
    train_dataset = ONTDataset(train_folder, reference_file, window_size, transform, max_number_windows)
    test_folder = "databases/natural_flappie_r941_native_ap_toy/test_reads"
    test_dataset = ONTDataset(test_folder, reference_file, window_size, transform)
    test_data = DataLoader(test_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_text2int_fn)
    sequence_length = window_size    

    # Model
    # Parameters
    TCN_parameters = {
        'n_layers': 2,
        'in_channels': 1,
        'out_channels': 10,
        'kernel_size': 3,
        'dropout': 0.8
    }
    LSTM_parameters = {
        'n_layers': 1,
        'sequence_length': sequence_length,
        'input_size': TCN_parameters['out_channels'], 
        'batch_size': batch_size, 
        'hidden_size': 2 * TCN_parameters['out_channels'],
        'dropout': 0.8,
        'bidirectional': True
    }
    decoder_parameters = {
        'initial_size': 2 * LSTM_parameters['hidden_size'],
        # The hidden size dim is always twice the initial_size
        'output_size': 5,  # n_classes: blank + 4 bases 
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'dropout': 0.8
    }
    # Create the model
    model = Network(TCN_parameters, LSTM_parameters, decoder_parameters)

    # Train the model
    # Training parameters
    training_parameters = {
        'algorithm': 'Hogwild',
        'n_processes': 10,
        'epochs': 1,
        'n_initialisation_epochs': 1,
        'batch_size': batch_size,
        'learning_rate': 5E-4,
        'max_learning_rate': 1E-2,
        'weight_decay': 1,
        'momemtum': 0.9,
        'optimiser': 'RMSprop',
        'sequence_length': sequence_length,
        'scheduler': 'OneCycleLR',
        'in_hpc': False
    }

    print('Model: ')
    print(model)

    # Training
    train(model, train_dataset, **training_parameters)

    # test = list(train_data)[0]
    # output = model(test['signals'])
    # print(output.shape)

    # # Decode the output
    # output_sequences = list(decoder(output, test['fragments']))
    # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    
            

    

    


    
    

    

    