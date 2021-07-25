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
from datasets import Dataset_ap, CombinedDataset, CustomisedSampler
from models import TCN_module, LSTM_module, DecoderChiron
from datetime import datetime
from training_utils import train
from datasets import collate_text2int_fn
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

    # database_dir = '/home/mario/Projects/project_2/databases/toy_working_ap'

    # Set fast5 and reference
    reference_file = database_dir + '/' + 'reference.fasta'

    # Transforms

    # Load the train and test datasets
    batch_size = 256
    window_sizes = [300]
    max_reads = 1000
    max_windows = None
    train_folder = database_dir + '/' + "reads"
    
    # # Load dataset
    # train_dataset_200 = Dataset_ap(train_folder, reference_file, window_sizes[0], max_windows, flowcell='flowcell1', hq_value='Q20', max_reads=max_reads)
    # train_dataset_400 = Dataset_ap(train_folder, reference_file, window_sizes[1], max_windows, flowcell='flowcell1', hq_value='Q20', max_reads=max_reads)
    # train_dataset_1000 = Dataset_ap(train_folder, reference_file, window_sizes[2], max_windows, flowcell='flowcell1', hq_value='Q20', max_reads=max_reads)
    # train_dataset = CombinedDataset(train_dataset_200, train_dataset_400, train_dataset_1000)

    # train_data = CustomisedDataLoader(dataset=train_dataset, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn)

    # train_dataset_300 = Dataset_ap(train_folder, reference_file, window_sizes[0], max_windows, hq_value='Q20', max_reads=max_reads)
    # train_data = DataLoader(train_dataset_300, batch_size=batch_size, shuffle=True, collate_fn=collate_text2int_fn)

    # # Model
    # # Parameters
    # TCN_parameters = {
    #     'n_layers': 5,
    #     'in_channels': 1,
    #     'out_channels': 256,
    #     'kernel_size': 3,
    #     'dropout': 0.8
    # }
    # LSTM_parameters = {
    #     'n_layers': 3,
    #     'input_size': TCN_parameters['out_channels'], 
    #     'batch_size': batch_size, 
    #     'hidden_size': 200, # 2 * TCN_parameters['out_channels'],
    #     'dropout': 0.8,
    #     'bidirectional': True
    # }
    # decoder_parameters = {
    #     'initial_size': 2 * LSTM_parameters['hidden_size'],
    #     # The hidden size dim is always twice the initial_size
    #     'output_size': 5,  # n_classes: blank + 4 bases
    #     'batch_size': batch_size,
    #     'dropout': 0.8
    # }
    
    # # Create the model
    # model = Network(TCN_parameters, LSTM_parameters, decoder_parameters)  

    # # Training parameters
    # training_parameters = {
    #     'algorithm': 'DataParallel',
    #     'n_processes': 1,
    #     'n_epochs': 1,
    #     'n_initialisation_epochs': 0,
    #     'batch_size': batch_size,
    #     'learning_rate': 0.0001,
    #     'max_learning_rate': 1E-2,
    #     'weight_decay': 0.01,
    #     'momemtum': 0,
    #     'optimiser': 'Adam',
    #     'sequence_lengths': window_sizes,
    #     'scheduler': 'OneCycleLR',
    #     'in_hpc': True,
    #     'max_batches': 500
    # }

    # # Print model architecture
    # print('Model: ')
    # print(model)

    # # Print training parameters
    # text_training = f"""
    # Training parameters:
    # - Algorithm: {training_parameters['algorithm']}
    # - N processes: {training_parameters['n_processes']}
    # - Epochs: {training_parameters['n_epochs']}
    # - N initialisation epochs: {training_parameters['n_initialisation_epochs']}
    # - Batch size: {training_parameters['batch_size']}
    # - Learning rate: {training_parameters['learning_rate']}
    # - Max learning rate: {training_parameters['max_learning_rate']}
    # - Weight decay: {training_parameters['weight_decay']}
    # - Momemtum: {training_parameters['momemtum']}
    # - Optimiser: {training_parameters['optimiser']}
    # - Sequence lengths (alternating): {training_parameters['sequence_lengths']}
    # - Scheduler: {training_parameters['scheduler']}
    # - In HPC: {training_parameters['in_hpc']}
    # - N Batches: {training_parameters['max_batches']}
    # """
    # print(text_training)

    # # Set up Comet
    # experiment_name = f"acinetobacter-train-{str(datetime.now()).replace(' ', '_')}"
    # experiment = Experiment(
    #     api_key="rqM9qXHiO7Ai4U2cqj1pS4R2R",
    #     project_name="project-2",
    #     workspace="mrubio-chavarria",
    # )
    # experiment.set_name(experiment_name)
    # experiment.display()

    # # Log training parameters
    # experiment.log_parameters({
    #     'algorithm': training_parameters['algorithm'],
    #     'n_epochs': training_parameters['n_epochs'],
    #     'n_initialisation_epochs': training_parameters['n_initialisation_epochs'],
    #     'batch_size': training_parameters['batch_size'],
    #     'learning_rate': training_parameters['learning_rate'],
    #     'max_learning_rate': training_parameters['max_learning_rate'],
    #     'weight_decay': training_parameters['weight_decay'],
    #     'momemtum': training_parameters['momemtum'],
    #     'optimiser': training_parameters['optimiser'],
    #     'sequence_lengths': training_parameters['sequence_lengths'],
    #     'scheduler': training_parameters['scheduler']
    # })
   
    # # # Training
    # # train(model, train_data, experiment, **training_parameters)
    
    # # # Save the model
    # # time = str(datetime.now()).replace(' ', '_')
    # # model_name = f'model_{time}.pt'
    # # model_path = database_dir + '/' + 'saved_models' + '/' + model_name
    # # torch.save(model.state_dict(), model_path)
    # # experiment.log_model(f'model_{time}', model_path)  #  Large uploading time

    # # test = list(train_data)[0]
    # # output = model(test['signals'])
    # # print(output.shape)

    # # # Decode the output
    # # output_sequences = list(decoder(output, test['fragments']))
    # # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    # # Stop recording parameters
    # experiment.end()

    
            

    

    


    
    

    

    