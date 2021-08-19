#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The code below solves the problem of correcting the basecalled
sequence based on the semi-global alignment with the reference
sequence.
"""

# Libraries
from typing_extensions import OrderedDict
from comet_ml import Experiment
import torch
import sys
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset_3xr6, Dataset_3xr6_transformed, Dataset_ap, collate_text2int_fn, CombinedDataset, CustomisedSampler
from models import DecoderChiron, DecoderCustom, ResidualBlockIV, TCN_module, LSTM_module, DecoderGELU
from bnlstm import LSTM
from datetime import datetime
from dataloaders import CombinedDataLoader, CustomisedDataLoader
from training_utils import train
from pytictoc import TicToc


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
        self.decoder = DecoderChiron(**self.decoder_parameters)

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    # Project directory
    # database_dir = sys.argv[2]

    # # Load previous model
    # model_path = "/home/mario/Documentos/Imperial/Project_2/output_experiments/experiment2/3xr6"
    # model_path += "/" + "model_2021-08-10_21:13:45.501383_4e226781-afc3-4bdd-958c-ad529e507bc3.pt"
    # checkpoint = torch.load(model_path) if model_path else None
    # os.environ['CHECKPOINT'] = checkpoint


    model = OrderedDict({
        "module.load": 1,
        "module.step": 2,
        "module.clash": 3
    })

    model = {key[7::]: value for key, value in model.items()}

    database_dir_ap = '/home/mario/Projects/project_2/databases/toy_working_ap'
    database_dir_3xr6 = '/home/mario/Projects/project_2/databases/toy_working_3xr6'

    # Storage file
    # file_manual_record = sys.argv[3]
    file_manual_record = "/home/mario/Projects/project_2/manual_record.tsv"

    # Set fast5 and reference
    reference_file_ap = database_dir_ap + '/' + 'reference.fasta'
    reference_file_3xr6 = database_dir_3xr6 + '/' + 'reference.fasta'

    batch_size = 32
    shuffle = True
    # Load the train dataset
    train_window_sizes = [200, 400, 1000]
    train_max_reads = None  # Select all the reads
    train_max_batches = 10000
    train_max_windows = int(train_max_batches * (batch_size + 1))
    train_folder_ap = database_dir_ap + '/' + 'train_reads'
    train_folder_3xr6 = database_dir_3xr6 + '/' + 'train_reads'
    
    # Load the test dataset
    validation_window_sizes = [300]
    validation_max_batches = 5
    validation_max_windows = int(validation_max_batches * (batch_size + 1) / 2)  # Controls test dataset size: 3 epoch
    validation_max_reads = 2000  # Select all the reads
    validation_folder_ap = database_dir_ap + '/' + 'validation_reads'
    validation_folder_3xr6 = database_dir_3xr6 + '/' + 'validation_reads'
    
    # Load dataset
    train_dataset_200_1 = Dataset_3xr6(train_folder_3xr6, reference_file_3xr6, train_window_sizes[0], int(train_max_windows / 2), hq_value='Q7', max_reads=train_max_reads, index=0)
    train_dataset_400 = Dataset_ap(train_folder_ap, reference_file_ap, train_window_sizes[1], train_max_windows, hq_value='Q7', max_reads=train_max_reads, index=1)
    train_dataset_1000 = Dataset_3xr6(train_folder_3xr6, reference_file_3xr6, train_window_sizes[2], train_max_windows, hq_value='Q7', max_reads=train_max_reads, index=2)
    train_dataset_200_2 = Dataset_ap(train_folder_3xr6, reference_file_ap, train_window_sizes[0], int(train_max_windows / 2), hq_value='Q7', max_reads=train_max_reads, index=0)
    train_dataset = CombinedDataset(train_dataset_200_1, train_dataset_400, train_dataset_1000, train_dataset_200_2)    

    validation_dataset_1 = Dataset_3xr6(validation_folder_3xr6, reference_file_3xr6, validation_window_sizes[0], validation_max_windows, hq_value='Q7', max_reads=train_max_reads, validation=True)
    validation_dataset_2 = Dataset_ap(validation_folder_ap, reference_file_ap, validation_window_sizes[0], validation_max_windows, hq_value='Q7', max_reads=train_max_reads, validation=True)
    validation_dataset = CombinedDataset(validation_dataset_1, validation_dataset_2)    

    train_data = CustomisedDataLoader(dataset=train_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn, shuffle=shuffle)
    validation_data = CustomisedDataLoader(dataset=validation_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn, shuffle=shuffle)
    # flowcell = 'flowcell1'
    # flowcell = None
    # shuffle = False
    # train_dataset_300 = Dataset_3xr6(train_folder, reference_file, window_sizes[0], max_windows, flowcell=flowcell, hq_value='Q20')
    # train_data = DataLoader(train_dataset_300, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_text2int_fn)

    # Model
    # Parameters
    TCN_parameters = {
        'n_layers': 1,
        'in_channels': 1,
        'out_channels': 256,
        'kernel_size': 3,
        'dropout': 0.8
    }
    LSTM_parameters = {
        'n_layers': 1,
        'input_size': TCN_parameters['out_channels'], 
        'batch_size': batch_size, 
        'hidden_size': 200, # 2 * TCN_parameters['out_channels'],
        'dropout': 0.6,
        'bidirectional': True,
        'batch_norm': True,
        'gamma': 0.2
    }
    decoder_parameters = {
        'initial_size': 2 * LSTM_parameters['hidden_size'],
        # The hidden size dim is always twice the initial_size
        'output_size': 5,  # n_classes: 4 bases  + blank
        'batch_size': batch_size,
        'dropout': 0.2
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

    # Create the model
    model = Network(TCN_parameters, LSTM_parameters, decoder_parameters)

    # Training parameters
    training_parameters = {
        'algorithm': 'single',
        'n_processes': 1,
        'n_epochs': 2,
        'n_initialisation_epochs': 0,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'max_learning_rate': 1E-2,
        'weight_decay': 0,
        'momemtum': 0,
        'optimiser': 'Adam',
        'sequence_lengths': train_window_sizes,
        'scheduler': None, #'StepLR',
        'in_hpc': True,
        'max_batches': 5,
        'step_size': 5,
        'n_labels': decoder_parameters['output_size'],
        'file_manual_record': file_manual_record
    }
    # Print model architecture
    print('Model: ')
    print(model)

    # Print training parameters
    text_training = f"""
    Training parameters:
    - Algorithm: {training_parameters['algorithm']}
    - N processes: {training_parameters['n_processes']}
    - Epochs: {training_parameters['n_epochs']}
    - N initialisation epochs: {training_parameters['n_initialisation_epochs']}
    - Batch size: {training_parameters['batch_size']}
    - Learning rate: {training_parameters['learning_rate']}
    - Max learning rate: {training_parameters['max_learning_rate']}
    - Weight decay: {training_parameters['weight_decay']}
    - Momemtum: {training_parameters['momemtum']}
    - Optimiser: {training_parameters['optimiser']}
    - Sequence lengths: {training_parameters['sequence_lengths']}
    - Scheduler: {training_parameters['scheduler']}
    - In HPC: {training_parameters['in_hpc']}
    - N Batches: {training_parameters['max_batches']}
    - File: {training_parameters['file_manual_record']}
    """
    print(text_training)
   
    # Training
    train(model, train_data, validation_data, **training_parameters)
    
    # # # Save the model
    # # time = str(datetime.now()).replace(' ', '_')
    # # model_name = f'model_{time}.pt'
    # # model_path = database_dir + '/' + 'saved_models' + '/' + model_name
    # # torch.save(model.state_dict(), model_path)
    # # # experiment.log_model(f'model_{time}', model_path)

    # # # test = list(train_data)[0]
    # # # output = model(test['signals'])
    # # # print(output.shape)

    # # # # Decode the output
    # # # output_sequences = list(decoder(output, test['fragments']))
    # # # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))


    
            

    

    


    
    

    

    