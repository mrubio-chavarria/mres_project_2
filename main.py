#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The code below solves the problem of correcting the basecalled
sequence based on the semi-global alignment with the reference
sequence.
"""

# Libraries
import torch
import sys
import os
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torchaudio
from datasets import Dataset_3xr6, Dataset_3xr6_transformed, RawONTDataset, PreONTDataset, reshape2Tensor, collate_text2int_fn, text2int
from models import ResidualBlockIV, TCN_module, LSTM_module, ClassifierGELU, ResNet
from metrics import cer, _levenshtein_distance, char_errors
from ont_fast5_api.fast5_interface import get_fast5_file
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


def launch_training(model, train_data, device, rank=0, sampler=None, **kwargs):
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
    # sequences_lengths = tuple([sequence_length] * batch_size)
    sequences_lengths = tuple([156] * batch_size)
    log_softmax = nn.LogSoftmax(dim=2).to(device)
    loss_function = nn.CTCLoss().to(device)
    initialisation_loss_function = nn.CrossEntropyLoss().to(device)
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
            batch, target = batch['signals'].to(device), batch['targets'].to(device)
            if batch.shape[0] != batch_size:
                continue
            # Forward pass
            output = model(batch)
            # Loss
            # Set different loss to initialise
            output_size = output.shape
            # if epoch in initialisation_epochs:
            if False:
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
                output_size = output.shape
                output = output.view(output_size[0], output_size[2], output_size[1])
                loss = initialisation_loss_function(output, targets)
            else:
                # Regular
                # Loss function: CTC
                # Compute the loss
                output = output.view(output_size[1], output_size[0], output_size[2])
                loss = loss_function(
                    log_softmax(output),
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
            output_sequences = list(decoder(output.view(*output_size), target_segments))
            error_rates = [cer(target_sequences[i], output_sequences[i]) for i in range(len(output_sequences))]
            avg_error = sum(error_rates) / len(error_rates)
            # Show progress
            print('----------------------------------------------------------------------------------------------------------------------')
            print(f'First target: {target_sequences[0]}\nFirst output: {output_sequences[0]}')
            # print(f'First target: {target_sequences[0]}\nFirst output: {seq}')
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
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training started')
    print('Algorithm:', algorithm)
    # Select training algorithm
    if algorithm == 'single':
        model.to(device)
        model.train()
        # Prepare the data
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_text2int_fn)
        # Start training
        launch_training(model, train_data, device, **kwargs)
    elif algorithm == 'DataParallel':
        # Prepare model and data
        model = nn.DataParallel(model)
        model.to(device)
        model.train()
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_text2int_fn)
        # Start training
        launch_training(model, train_data, device, **kwargs)

    elif algorithm == 'Hogwild':
        # Start training
        # We are not setting a blockade per epoch
        model.to(device)
        model.train()
        model.share_memory()
        processes = []
        for rank in range(n_processes):
            train_sampler = DistributedSampler(train_dataset, num_replicas=n_processes, rank=rank)
            train_data = DataLoader(dataset=train_dataset,
                            sampler=train_sampler,
                            batch_size=kwargs['batch_size'],
                            collate_fn=collate_text2int_fn)
            
            print(f'Process {rank} launched')
            process = mp.Process(target=launch_training, args=(model, train_data, device, rank, train_sampler), kwargs=kwargs)
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

    # Set cuda devices visible
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    # Project directory
    # database_dir = '/home/mario/Projects/project_2/databases/working_3xr6'
    database_dir = sys.argv[2]

    # Set fast5 and reference
    reference_file = database_dir + '/' + 'reference.fasta'

    # Transforms
    transform = reshape2Tensor((1, -1))
    # Mel Spectrogram
    sample_rate = 4000
    n_fft = 100
    window_length = n_fft
    hop_length = 1
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length
    )
    # Pack the transforms together
    transform = nn.Sequential(
        mel_spectrogram
    )

    # Load the train and test datasets
    batch_size = 1
    window_size = 311
    max_windows = 300
    train_folder = database_dir + '/' + "reads"
    # test_folder = project_dir + '/' + "databases/natural_flappie_r941_native_ap_toy/test_reads"
    
    # train_dataset = Dataset_3xr6_transformed(train_folder, reference_file, window_size, max_windows, transform)
    train_dataset = Dataset_3xr6(train_folder, reference_file, window_size, max_windows, 'flowcell3')

    # Model
    # Parameters
    sequence_length = window_size
    TCN_parameters = {
        'n_layers': 1,
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 3,
        'dropout': 0.8
    }
    LSTM_parameters = {
        'n_layers': 1,
        'sequence_length': sequence_length,
        'input_size': TCN_parameters['out_channels'], 
        'batch_size': batch_size, 
        'hidden_size': 512, # 2 * TCN_parameters['out_channels'],
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
    output = model(torch.unsqueeze(train_dataset[0]['signal'], dim=0))
    
    print('Output:')
    print(output)
    

    # # Training parameters
    # training_parameters = {
    #     'algorithm': 'DataParallel',
    #     'n_processes': 1,
    #     'epochs': 1,
    #     'n_initialisation_epochs': 1,
    #     'batch_size': batch_size,
    #     'learning_rate': 5E-4,
    #     'max_learning_rate': 1E-2,
    #     'weight_decay': 1,
    #     'momemtum': 0.9,
    #     'optimiser': 'RMSprop',
    #     'sequence_length': sequence_length,
    #     'scheduler': 'OneCycleLR',
    #     'in_hpc': True
    # }

    # print('Model: ')
    # print(model)

    # text_training = f"""
    # Training parameters:
    # - Algorithm: {training_parameters['algorithm']}
    # - N processes: {training_parameters['n_processes']}
    # - Epochs: {training_parameters['epochs']}
    # - N initialisation epochs: {training_parameters['n_initialisation_epochs']}
    # - Batch size: {training_parameters['batch_size']}
    # - Learning rate: {training_parameters['learning_rate']}
    # - Max learning rate: {training_parameters['max_learning_rate']}
    # - Weight decay: {training_parameters['weight_decay']}
    # - Momemtum: {training_parameters['momemtum']}
    # - Optimiser: {training_parameters['optimiser']}
    # - Sequence length: {training_parameters['sequence_length']}
    # - Scheduler: {training_parameters['scheduler']}
    # - In HPC: {training_parameters['in_hpc']}
    # """
    # print(text_training)

    # # Training
    # train(model, train_dataset, **training_parameters)

    # test = list(train_data)[0]
    # output = model(test['signals'])
    # print(output.shape)

    # # Decode the output
    # output_sequences = list(decoder(output, test['fragments']))
    # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    
            

    

    


    
    

    

    