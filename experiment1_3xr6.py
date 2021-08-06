#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This scripts executes the experiment on the acinetobacter
dataset.
"""

# Libraries
import os 
import sys
import torch
from torch import nn
from datasets import Dataset_3xr6
from models import DecoderChiron, TCN_module, LSTM_module
from datetime import datetime
from training_utils import train
from datasets import collate_text2int_fn, CombinedDataset, CustomisedSampler
from uuid import uuid4
from dataloaders import CustomisedDataLoader
from torch.utils.data import DataLoader


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
    """
    # Set cuda devices visible
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    # Project directory
    database_dir = sys.argv[2]

    # Storage file
    file_manual_record = sys.argv[3]

    # Storage file
    gamma_value = (float(int(sys.argv[4])) - 1) / 10

    # Set fast5 and reference
    reference_file = database_dir + '/' + 'reference.fasta'

    batch_size = 32
    shuffle = False
    # Load the train dataset
    train_window_sizes = [200, 400, 1000]
    train_max_reads = 668  # Select all the reads
    train_max_batches = 500
    train_max_windows = int(train_max_batches * (batch_size + 1))
    train_folder = database_dir + '/' + 'train_reads'
    
    # Load the test dataset
    validation_window_sizes = [300]
    validation_max_batches = 5
    validation_max_windows = int(validation_max_batches * (batch_size + 1))  # Controls test dataset size: 3 epoch
    validation_max_reads = 20  # Select all the reads
    validation_folder = database_dir + '/' + 'validation_reads'
    
    # Load dataset
    train_dataset_200 = Dataset_3xr6(train_folder, reference_file, train_window_sizes[0], train_max_windows, hq_value='Q7', max_reads=train_max_reads, index=0)
    train_dataset_400 = Dataset_3xr6(train_folder, reference_file, train_window_sizes[1], train_max_windows, hq_value='Q7', max_reads=train_max_reads, index=1)
    train_dataset_1000 = Dataset_3xr6(train_folder, reference_file, train_window_sizes[2], train_max_windows, hq_value='Q7', max_reads=train_max_reads, index=2)
    train_dataset = CombinedDataset(train_dataset_200, train_dataset_400, train_dataset_1000)

    validation_dataset_300 = Dataset_3xr6(validation_folder, reference_file, validation_window_sizes[0], validation_max_windows, hq_value='Q7', max_reads=train_max_reads, validation=True)
    validation_dataset = CombinedDataset(validation_dataset_300)

    train_data = CustomisedDataLoader(dataset=train_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn)
    validation_data = CustomisedDataLoader(dataset=validation_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn)
    validation_data = DataLoader(validation_dataset_300, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_text2int_fn)

    # Model
    # Parameters
    TCN_parameters = {
        'n_layers': 5,
        'in_channels': 1,
        'out_channels': 256,
        'kernel_size': 3,
        'dropout': 0.8
    }
    LSTM_parameters = {
        'n_layers': 3,
        'input_size': TCN_parameters['out_channels'], 
        'batch_size': batch_size, 
        'hidden_size': 200, # 2 * TCN_parameters['out_channels'],
        'dropout': 0.8,
        'bidirectional': True,
        'batch_norm': True if gamma_value > 0.0 else False,
        'gamma': gamma_value
    }
    decoder_parameters = {
        'initial_size': 2 * LSTM_parameters['hidden_size'],
        # The hidden size dim is always twice the initial_size
        'output_size': 5,  # n_classes: blank + 4 bases 
        'batch_size': batch_size,
        'dropout': 0.8
    }
    
    # Create the model
    model = Network(TCN_parameters, LSTM_parameters, decoder_parameters)  

    # Training parameters
    training_parameters = {
        'algorithm': 'DataParallel',
        'n_processes': 1,
        'n_epochs': 1,
        'n_initialisation_epochs': 0,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'max_learning_rate': 1E-2,
        'weight_decay': 0,
        'momemtum': 0,
        'optimiser': 'Adam',
        'sequence_lengths': train_window_sizes,
        'scheduler': None, #'OneCycleLR',
        'in_hpc': True,
        'max_batches': train_max_batches,
        'n_labels': decoder_parameters['output_size'],
        'shuffle': shuffle,
        'file_manual_record': file_manual_record
    }

    # Generate experiment ID
    experiment_id = str(uuid4())
    # Print ID
    print('****************************************************************')
    print(f'EXPERIMENT ID: {experiment_id}')
    print('****************************************************************')

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
    - Sequence lengths (alternating): {training_parameters['sequence_lengths']}
    - Scheduler: {training_parameters['scheduler']}
    - In HPC: {training_parameters['in_hpc']}
    - N Batches: {training_parameters['max_batches']}
    - Gamma value: {LSTM_parameters['gamma']}
    - BNLSTM: {LSTM_parameters['batch_norm']}
    """
    print(text_training)

    # Training
    train(model, train_data, validation_data, **training_parameters)
    
    # Save the model
    time = str(datetime.now()).replace(' ', '_')
    model_name = f'model_{time}_{experiment_id}.pt'
    model_path = database_dir + '/' + 'saved_models' + '/' + model_name
    torch.save(model.state_dict(), model_path)


    
            

    

    


    
    

    

    