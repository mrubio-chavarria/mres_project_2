#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This scripts executes the experiment on the acinetobacter
dataset.
"""

# Libraries
from comet_ml import Experiment
from dataloaders import CustomisedDataLoader
import os 
import sys
import torch
from torch import nn
from datasets import CustomisedSampler, Dataset_3xr6, CombinedDataset
from models import TCN_module, LSTM_module, DecoderChiron
from datetime import datetime
from training_utils import train
from datasets import collate_text2int_fn
from torch.utils.data import DataLoader
from uuid import uuid4


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

    # Configuration
    conf_index = int(sys.argv[4])

    # Set fast5 and reference
    reference_file = database_dir + '/' + 'reference.fasta'

    # Transforms

    # Load the train and test datasets
    batch_size = 32
    # window_sizes = [200, 400, 1000]
    window_sizes = [300]
    max_windows = None
    max_reads = None  # Select all the reads
    train_folder = database_dir + '/' + "reads"
    
    # Load dataset
    # train_dataset_200 = Dataset_3xr6(train_folder, reference_file, window_sizes[0], max_windows, hq_value='Q20')
    # train_dataset_400 = Dataset_3xr6(train_folder, reference_file, window_sizes[1], max_windows, hq_value='Q20')
    # train_dataset_1000 = Dataset_3xr6(train_folder, reference_file, window_sizes[2], max_windows, hq_value='Q20')
    # train_dataset = CombinedDataset(train_dataset_200, train_dataset_400, train_dataset_1000)


    # train_data = CustomisedDataLoader(dataset=train_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn)

    flowcell = 'flowcell1'
    train_dataset_300 = Dataset_3xr6(train_folder, reference_file, window_sizes[0], max_windows, flowcell=flowcell, hq_value='Q20')
    train_data = DataLoader(train_dataset_300, batch_size=batch_size, shuffle=True, collate_fn=collate_text2int_fn)

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
        'bidirectional': True
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
    configurations = [
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'max_learning_rate': 1E-2,
            'weight_decay': 0,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': None,
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        },
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'max_learning_rate': 1E-2,
            'weight_decay': 0.01,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': None,
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        },
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.0001,
            'max_learning_rate': 1E-2,
            'weight_decay': 0,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': None,
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        },
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.0001,
            'max_learning_rate': 1E-2,
            'weight_decay': 0.01,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': None,
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        },
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.01,
            'max_learning_rate': 1E-2,
            'weight_decay': 0,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': 'StepLR',
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        },
        {
            'algorithm': 'DataParallel',
            'n_processes': 1,
            'n_epochs': 5,
            'n_initialisation_epochs': 0,
            'batch_size': batch_size,
            'learning_rate': 0.01,
            'max_learning_rate': 1E-2,
            'weight_decay': 0.01,
            'momemtum': 0,
            'optimiser': 'Adam',
            'sequence_lengths': window_sizes,
            'scheduler': 'StepLR',
            'step_size': 2,
            'in_hpc': True,
            'max_batches': 500,
            'file_manual_record': file_manual_record
        }
    ]
    training_parameters = configurations[conf_index]


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
    """
    print(text_training)

    # Set up Comet
    record_experiment = False
    if record_experiment:
        experiment_name = f"acinetobacter-train-{str(datetime.now()).replace(' ', '_')}"
        experiment = Experiment(
            api_key="rqM9qXHiO7Ai4U2cqj1pS4R2R",
            project_name="project-2",
            workspace="mrubio-chavarria",
        )
        experiment.set_name(experiment_name)
        experiment.display()

        # Log training parameters
        experiment.log_parameters({
            'algorithm': training_parameters['algorithm'],
            'n_epochs': training_parameters['n_epochs'],
            'n_initialisation_epochs': training_parameters['n_initialisation_epochs'],
            'batch_size': training_parameters['batch_size'],
            'learning_rate': training_parameters['learning_rate'],
            'max_learning_rate': training_parameters['max_learning_rate'],
            'weight_decay': training_parameters['weight_decay'],
            'momemtum': training_parameters['momemtum'],
            'optimiser': training_parameters['optimiser'],
            'sequence_lengths': training_parameters['sequence_lengths'],
            'scheduler': training_parameters['scheduler']
        })
    else:
        experiment = None
    # Training
    train(model, train_data, experiment, **training_parameters)
    
    # Save the model
    time = str(datetime.now()).replace(' ', '_')
    model_name = f'model_{time}_{experiment_id}.pt'
    model_path = database_dir + '/' + 'saved_models' + '/' + model_name
    torch.save(model.state_dict(), model_path)
    # experiment.log_model(f'model_{time}', model_path)  #  Large uploading time

    # # test = list(train_data)[0]
    # # output = model(test['signals'])
    # # print(output.shape)

    # # # Decode the output
    # # output_sequences = list(decoder(output, test['fragments']))
    # # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    # Stop recording parameters
    if record_experiment:
        experiment.end()

    
            

    

    


    
    

    

    