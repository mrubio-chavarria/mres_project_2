
# Libraries
import torch
import pandas as pd
import os
import sys
from torch import nn
from fast_ctc_decode import beam_search, viterbi_search
from models import DecoderChiron, TCN_module, LSTM_module
from datasets import Dataset_3xr6, Dataset_ap, CombinedDataset, collate_text2int_fn, CustomisedSampler
from dataloaders import CustomisedDataLoader
from metrics import cer


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


def decoder(probabilities_matrix, n_labels=5, method='greedy'):
    """
    DESCRIPTION:
    The function that implements the greedy algorithm to obtain the
    sequence of letters from the probability matrix.
    :param probabilities_matrix: [torch.Tensor] matrix of dimensions
    [batch_size, sequence_length] with the output probabilities.
    :yield: [str] the sequence associated with a series of 
    probabilities.
    """
    letters = ['A', 'T', 'G', 'C', '$'][0:n_labels]
    # windows = [length2indices(window) for window in segments_lengths]
    if method == 'greedy':
        max_probabilities = torch.argmax(probabilities_matrix, dim=2)
        for i in range(len(probabilities_matrix)):
            # Output probabilities to sequence
            final_sequence = []
            sequence = [letters[prob] for prob in max_probabilities[i].tolist()]
            final_sequence = []
            for item in sequence:
                if not final_sequence:
                    final_sequence.append(item)
                    continue
                if final_sequence[-1] != item:
                    final_sequence.append(item)
            final_sequence = ''.join(final_sequence)
            final_sequence_greedy = ''.join([''.join(set(fragment)) for fragment in final_sequence.split('$')])
            yield final_sequence_greedy
    elif method == 'viterbi_search':
        probs = probabilities_matrix.cpu().detach().numpy()
        for prob in probs:
            seq, _ = viterbi_search(prob, ''.join(letters))
            yield seq.replace('$', '')
    elif method == 'beam_search':
        probs = probabilities_matrix.cpu().detach().numpy()
        for prob in probs:
            try:
                seq, _ = beam_search(prob, ''.join(letters), beam_size=20, beam_cut_threshold=1E-48)
                yield seq.replace('$', '')
            except:
                yield 'No good transcription'


if __name__ == "__main__":
    
    # Set cuda devices visible
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    # Checkpoint routes
    route_ap = "/rds/general/user/mr820/home/project_2/final_models/ap_model.pt"
    route_3xr6 = "/rds/general/user/mr820/home/project_2/final_models/3xr6_model.pt"
    route_both = "/rds/general/user/mr820/home/project_2/final_models/both_model.pt"

    # Load checkpoints
    checkpoint_ap = torch.load(route_ap) 
    checkpoint_3xr6 = torch.load(route_3xr6) 
    checkpoint_both = torch.load(route_both) 

    checkpoint_ap['model_state_dict'] = {key[7::]: value for key, value in checkpoint_ap['model_state_dict'].items()}
    checkpoint_3xr6['model_state_dict'] = {key[7::]: value for key, value in checkpoint_3xr6['model_state_dict'].items()}
    checkpoint_both['model_state_dict'] = {key[7::]: value for key, value in checkpoint_both['model_state_dict'].items()}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create and load the models
    # Parameters
    batch_size = 5
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
        'batch_norm': False,
        'gamma': 0.0
    }
    decoder_parameters = {
        'initial_size': 2 * LSTM_parameters['hidden_size'],
        # The hidden size dim is always twice the initial_size
        'output_size': 5,  # n_classes: blank + 4 bases 
        'batch_size': batch_size,
        'dropout': 0.8
    }

    # Create the model
    model_ap = Network(TCN_parameters, LSTM_parameters, decoder_parameters).to(device)  
    model_3xr6 = Network(TCN_parameters, LSTM_parameters, decoder_parameters).to(device)
    model_both = Network(TCN_parameters, LSTM_parameters, decoder_parameters).to(device)
    
    model_ap.load_state_dict(checkpoint_ap['model_state_dict'])
    model_3xr6.load_state_dict(checkpoint_3xr6['model_state_dict'])
    model_both.load_state_dict(checkpoint_both['model_state_dict'])

    # Load data
    # AP
    reference_file_ap = "/rds/general/user/mr820/home/project_2/databases/working_ap/reference.fasta"
    validation_folder_ap = "/rds/general/user/mr820/home/project_2/databases/working_ap/validation_reads"
    windows_300_1 = Dataset_ap(validation_folder_ap, reference_file_ap, 300, 500, hq_value='Q7', max_reads=3, index=0, validation=False)
    # 3xr6
    reference_file_3xr6 = "/rds/general/user/mr820/home/project_2/databases/working_3xr6/reference.fasta"
    validation_folder_3xr6 = "/rds/general/user/mr820/home/project_2/databases/working_3xr6/validation_reads"
    windows_300_2 = Dataset_3xr6(validation_folder_3xr6, reference_file_3xr6, 300, 500, hq_value='Q7', max_reads=300, validation=False)
    # Both
    windows_300_3 = Dataset_3xr6(validation_folder_3xr6, reference_file_3xr6, 300, 250, hq_value='Q7', max_reads=300, index=0)
    windows_300_4 = Dataset_ap(validation_folder_ap, reference_file_ap, 300, 250, hq_value='Q7', max_reads=3, index=0)
    windows_300_5 = CombinedDataset(windows_300_3, windows_300_4)

    shuffle = False
    test_data_ap = CustomisedDataLoader(dataset=windows_300_1, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn, shuffle=shuffle)
    test_data_3xr6 = CustomisedDataLoader(dataset=windows_300_2, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn, shuffle=shuffle)
    test_data_both = CustomisedDataLoader(dataset=windows_300_5, batch_size=batch_size, sampler=CustomisedSampler, collate_fn=collate_text2int_fn, shuffle=shuffle)

    # Set models for testing
    model_ap.eval()
    model_3xr6.eval()
    model_both.eval()

    # Prepare loss if needed
    log_softmax = nn.LogSoftmax(dim=2).to(device)

    # Go through the test batches
    losses = []
    errors = []
    datasets = []
    models = []
    n_labels = 5
    loss_function = nn.CTCLoss(blank=4).to(device)

    # Test AP
    for batch in test_data_ap:
        # Clean gradient
        model_ap.zero_grad()
        # Move data to device
        target_segments = batch['fragments']
        target_sequences = batch['sequences']
        targets_lengths = batch['targets_lengths']
        batch, target = batch['signals'].to(device), batch['targets'].to(device)
        # All the sequences in a batch are of the same length
        sequences_lengths = tuple([batch.shape[-1]] * batch.shape[0])
        # -----------------------------------------------------
        # AP model
        # -----------------------------------------------------
        # Forward pass
        output = model_ap(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('AP')
        models.append('AP')
        # -----------------------------------------------------
        # 3xr6 model
        # -----------------------------------------------------
        # Forward pass
        output = model_3xr6(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('AP')
        models.append('3xr6')
        # -----------------------------------------------------
        # Both model
        # -----------------------------------------------------
        # Forward pass
        output = model_both(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('AP')
        models.append('Both')

    # Test 3xr6
    for batch in test_data_3xr6:
        # Clean gradient
        model_ap.zero_grad()
        # Move data to device
        target_segments = batch['fragments']
        target_sequences = batch['sequences']
        targets_lengths = batch['targets_lengths']
        batch, target = batch['signals'].to(device), batch['targets'].to(device)
        # All the sequences in a batch are of the same length
        sequences_lengths = tuple([batch.shape[-1]] * batch.shape[0])
        # -----------------------------------------------------
        # AP model
        # -----------------------------------------------------
        # Forward pass
        output = model_ap(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('3xr6')
        models.append('AP')
        # -----------------------------------------------------
        # 3xr6 model
        # -----------------------------------------------------
        # Forward pass
        output = model_3xr6(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('3xr6')
        models.append('3xr6')
        # -----------------------------------------------------
        # Both model
        # -----------------------------------------------------
        # Forward pass
        output = model_both(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('3xr6')
        models.append('Both')

    # Test 3xr6
    for batch in test_data_both:
        # Clean gradient
        model_ap.zero_grad()
        # Move data to device
        target_segments = batch['fragments']
        target_sequences = batch['sequences']
        targets_lengths = batch['targets_lengths']
        batch, target = batch['signals'].to(device), batch['targets'].to(device)
        # All the sequences in a batch are of the same length
        sequences_lengths = tuple([batch.shape[-1]] * batch.shape[0])
        # -----------------------------------------------------
        # AP model
        # -----------------------------------------------------
        # Forward pass
        output = model_ap(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('Both')
        models.append('AP')
        # -----------------------------------------------------
        # 3xr6 model
        # -----------------------------------------------------
        # Forward pass
        output = model_3xr6(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('Both')
        models.append('3xr6')
        # -----------------------------------------------------
        # Both model
        # -----------------------------------------------------
        # Forward pass
        output = model_both(batch)
        # Decode output
        output_sequences = list(decoder(output, n_labels))
        error_rates = [cer(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        # Compute the loss
        loss = loss_function(
            log_softmax(output.permute(1, 0, 2)),
            target,
            sequences_lengths,
            targets_lengths
        )
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
        datasets.append('Both')
        models.append('Both')

    # Store data
    data = {
        'loss': losses,
        'error': errors,
        'dataset': datasets,
        'model': models
    }
    file = "/rds/general/user/mr820/home/project_2/test_data.tsv"
    pd.DataFrame.from_dict(data).to_csv(file, sep='\t', header=False)