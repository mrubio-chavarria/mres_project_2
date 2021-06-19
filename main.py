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
from datasets import ONTDataset, reshape2Tensor, collate_text2int_fn
from models import ResidualBlockIV, TCN_module, LSTM_module, ClassifierGELU, ResNet
from metrics import cer, _levenshtein_distance, char_errors
from ont_fast5_api.fast5_interface import get_fast5_file
import os


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
    letters = ['', 'A', 'T', 'G', 'C']
    windows = [length2indices(window) for window in segments_lengths]
    for i in range(len(probabilities_matrix)):
        # Output probabilities to sequence
        sequence = [[letters[prob] for prob in max_probabilities[i].tolist()][windows[i][index]:windows[i][index+1]]
            for index in range(len(windows[i])-1)]
        sequence = ''.join([''.join(list(set(segment))) for segment in sequence])
        yield sequence


def train(model, loss_function, sequence_length, optimiser, scheduler):
    """
    DESCRIPTION:
    A function to launch the model training.
    :param model: [nn.Module] the model to train.
    :param loss_function: [nn.Module] the function to compute the loss.
    :param sequence_length: [int] the length of every sequence, constant parameter.
    :param optimiser: [torch.optim] the optimiser to descend in the gradient.
    :param scheduler: [torch.optim.lr_scheduler] scheduler to control the adaptive 
    learning rate.
    """
    model.train()
    sequences_lengths = tuple([sequence_length] * batch_size)
    for epoch in range(epochs):
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
            loss = loss_function(
                output.reshape(sequence_length, batch_size, -1),
                target,
                sequences_lengths,
                targets_lengths
            )
            # Backward pass
            loss.backward()
            # Gradient step
            optimiser.step()
            scheduler.step()
            # Decode output
            output_sequences = list(decoder(output, target_segments))
            error_rates = [cer(target_sequences[i], output_sequences[i]) for i in range(len(output_sequences))]
            avg_error = sum(error_rates) / len(error_rates)
            # Show progress
            print('-----------------------------------------------------------------------------------------------')
            print(f'First target: {target_sequences[0]}\nFirst output: {output_sequences[0]}')
            print(f'Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error} Learning rate: {optimiser.param_groups[0]["lr"]}')


if __name__ == "__main__":
    
    """
    NOTES TO CONSIDER:
    - The data has not been rescaled, although Tombo normalised the signal.
    It is not the same.
    """

    # Set fast5 and reference
    # reads_folder = "databases/synthetic_flappie_r941_native_3xr6/reads"
    reference_file = "databases/natural_flappie_r941_native_ap_toy/reference.fasta"
    
    # Load the train and test datasets
    transform = reshape2Tensor((1, -1))
    batch_size = 64
    window_size = 311
    train_folder = "databases/natural_flappie_r941_native_ap_toy/train_reads"
    train_dataset = ONTDataset(train_folder, reference_file, window_size, transform)
    train_data = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_text2int_fn)
    test_folder = "databases/natural_flappie_r941_native_ap_toy/test_reads"
    test_dataset = ONTDataset(test_folder, reference_file, window_size, transform)
    test_data = DataLoader(test_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_text2int_fn)
    sequence_length = window_size

    # Model
    # Parameters
    batch_size = 64
    TCN_parameters = {
        'n_layers': 5,
        'in_channels': 1,
        'out_channels': 50,
        'kernel_size': 3,
        'dropout': 0.5
    }
    LSTM_parameters = {
        'n_layers': 2,
        'sequence_length': sequence_length,
        'input_size': TCN_parameters['out_channels'], # 512 
        'batch_size': batch_size, 
        'hidden_size': 2*TCN_parameters['out_channels'], 
        'output_size': 5,
        'dropout': 0.8,
        'bidirectional': True
    }
    decoder_parameters = {
        'initial_size': LSTM_parameters['hidden_size'],
        'hidden_size': LSTM_parameters['hidden_size'],
        'output_size': 5,  # n_classes: 4 bases + space
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'dropout': 0.8

    }
    # Create the model
    model = Network(TCN_parameters, LSTM_parameters, decoder_parameters)
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train the model
    # Parameters
    epochs = 1
    loss_function = nn.CTCLoss(blank=0).to(device)
    learning_rate = 1E-3
    max_learning_rate = 1E-1
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
        max_lr=max_learning_rate,
        epochs=epochs,
        steps_per_epoch=int(len(train_data)),
        anneal_strategy='linear')
    # Training
    train(model, loss_function, sequence_length, optimiser, scheduler)


    # test = list(train_data)[0]
    # output = model(test['signals'])
    # print(output.shape)

    # # Decode the output
    # output_sequences = list(decoder(output, test['fragments']))
    # errors = [cer(test['sequences'][i], output_sequences[i]) for i in range(len(test['sequences']))]
    # # print(loss_function(output.view(sequence_length, batch_size, -1), test['targets'], sequences_lengths, test['targets_lengths']))

    
            

    

    


    
    

    

    