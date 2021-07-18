#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This scripts executes the experiment on the acinetobacter
dataset.
"""

# Libraries
from comet_ml import Experiment
import os 
import sys
import torch
from torch import nn
import torchaudio
from datasets import Dataset_3xr6
from models import TCN_module, LSTM_module, ClassifierGELU
from datetime import datetime
from training_utils import train


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

    # database_dir = '/home/mario/Projects/project_2/databases/toy_working_ap'

    # Set fast5 and reference
    reference_file = database_dir + '/' + 'reference.fasta'

    # Transforms
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
    batch_size = 32
    window_size = 311
    max_windows = None
    train_folder = database_dir + '/' + "reads"
    # test_folder = project_dir + '/' + "databases/natural_flappie_r941_native_ap_toy/test_reads"
    
    # train_dataset = Dataset_3xr6_transformed(train_folder, reference_file, window_size, max_windows, transform)
    train_dataset = Dataset_3xr6(train_folder, reference_file, window_size, max_windows, hq_value='Q20')

    # Model
    # Parameters
    sequence_length = window_size
    TCN_parameters = {
        'n_layers': 3,
        'in_channels': 1,
        'out_channels': 32,
        'kernel_size': 3,
        'dropout': 0.8
    }
    LSTM_parameters = {
        'n_layers': 5,
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

    # Training parameters
    training_parameters = {
        'algorithm': 'DataParallel',
        'n_processes': 1,
        'n_epochs': 1,
        'n_initialisation_epochs': 0,
        'batch_size': batch_size,
        'learning_rate': 5E-4,
        'max_learning_rate': 1E-2,
        'weight_decay': 1,
        'momemtum': 0.9,
        'optimiser': 'Adam',
        'sequence_length': sequence_length,
        'scheduler': None, # 'OneCycleLR',
        'in_hpc': True
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
    - Sequence length: {training_parameters['sequence_length']}
    - Scheduler: {training_parameters['scheduler']}
    - In HPC: {training_parameters['in_hpc']}
    """
    print(text_training)

    # Set up Comet
    experiment_name = f"3xr6-train-{str(datetime.now()).replace(' ', '_')}"
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
        'sequence_length': training_parameters['sequence_length'],
        'scheduler': training_parameters['scheduler']
    })
   
    # Training
    train(model, train_dataset, experiment, **training_parameters)
    
    # Save the model
    time = str(datetime.now()).replace(' ', '_')
    model_name = f'model_{time}.pt'
    model_path = database_dir + '/' + 'saved_models' + '/' + model_name
    torch.save(model.state_dict(), model_path)
    experiment.log_model(f'model_{time}', model_path)

    # test = list(train_data)[0]
    # output = model(test['signals'])
    # print(output.shape)

    # # Decode the output
    # output_sequences = list(decoder(output, test['fragments']))
    # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    # Stop recording parameters
    experiment.end()

    
            

    

    


    
    

    

    