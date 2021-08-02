#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The functions in this file develop the training in the 
experiment scripts.
"""

# Libraries
from losses import FocalCTCLoss
import torch
from torch import log_softmax, nn
from metrics import cer
import pandas as pd
from fast_ctc_decode import beam_search, viterbi_search
from torch.optim.lr_scheduler import StepLR
from pytictoc import TicToc


# Functions
def length2indices(window):
    indeces = [0]
    for i in range(len(window)):
        indeces.append(window[i] + indeces[i])
    return indeces


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


def launch_training(model, train_data, validation_data, device, **kwargs):
    """
    DESCRIPTION:
    Function that effectively launches the training. 
    :param model: [nn.Module] the model to train.
    :param train_data: [iter] the data organised in batches ready to train.
    :param test_data: [iter] the data organised in batches ready to test.
    :param device: [torch.device] device to move the data into.
    """
    # Helper functions
    def record_in_file(train_loss_values, train_avgcer_values, test_loss_values, test_avgcer_values):
        """
        DESCRIPTION:
        Helper function to store all the information in the same file.
        :param train_loss_values: [list] train loss values to store.
        :param train_avgcer_values: [list] train average cer values to store.
        :param train_loss_values: [list] test loss values to store.
        :param train_avgcer_values: [list] test average cer values to store.
        """
        # Read file
        file = kwargs.get('file_manual_record')
        # Format train and test data in same table
        loss_values = train_loss_values + test_loss_values
        avgcer_values = train_avgcer_values + test_avgcer_values
        training = [True] * len(train_loss_values) + [False] * len(test_loss_values)
        data = {'loss': loss_values, 'avgcer': avgcer_values, 'training': training}
        # Store
        pd.DataFrame.from_dict(data).to_csv(file, sep='\t', header=False)
    
    # Create optimiser
    if kwargs.get('optimiser', 'SGD') == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-3),
                                    momentum=kwargs.get('momemtum', 0))
    elif kwargs.get('optimiser', 'SGD') == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-3),
                                    weight_decay=kwargs.get('weight_decay', 0))
    elif kwargs.get('optimiser', 'SGD') == 'AdamW':
        optimiser = torch.optim.AdamW(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-3),
                                    weight_decay=kwargs.get('weight_decay', 0))
    elif kwargs.get('optimiser', 'SGD') == 'RMSprop':
        optimiser = torch.optim.RMSprop(model.parameters(),
                                        lr=kwargs.get('learning_rate', 1E-3),
                                        weight_decay=kwargs.get('weight_decay',0),
                                        momentum=kwargs.get('momemtum', 0))
    else:
        raise ValueError('Invalid optimiser selected')
    # Max number of batches
    print('Optimiser:', optimiser)
    max_batches = kwargs.get('max_batches', None)
    # Create scheduler
    if kwargs.get('scheduler', None) is not None:
        if kwargs.get('scheduler', None) == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
                                                            max_lr=kwargs.get('max_learning_rate', 1E-2),
                                                            steps_per_epoch=len(train_data),
                                                            epochs=kwargs.get('n_epochs', 30))
        elif kwargs.get('scheduler', None) == 'StepLR':
            scheduler = StepLR(optimiser, step_size=kwargs.get('step_size', 5), gamma=0.1)

    # Prepare training
    batch_size = kwargs.get('batch_size')
    # sequences_lengths = tuple([sequence_length] * batch_size)
    log_softmax = nn.LogSoftmax(dim=2).to(device)
    loss_function = FocalCTCLoss(blank=4).to(device)
    loss_function = nn.CTCLoss(blank=4).to(device)
    initialisation_loss_function = nn.CrossEntropyLoss().to(device)
    initialisation_epochs = range(kwargs.get('n_initialisation_epochs', 1))
    # Train
    train_loss = []
    train_avgcer = []
    test_loss = []
    test_avgcer = []
    for epoch in range(kwargs.get('n_epochs', 5)):
        for batch_id, batch in enumerate(train_data):
            if max_batches is not None:
                if batch_id == max_batches:
                    break
            # Clean gradient
            model = model.to(device)
            model.zero_grad()
            # Move data to device
            target_segments = batch['fragments']
            target_sequences = batch['sequences']
            targets_lengths = batch['targets_lengths']
            batch, target = batch['signals'].to(device), batch['targets'].to(device)
            # All the sequences in a batch are of the same length
            sequences_lengths = tuple([batch.shape[-1]] * batch.shape[0])  
            if batch.shape[0] != batch_size:
                continue
            # Forward pass
            output = model(batch)
            # Decode output
            output_sequences = list(decoder(output, kwargs.get('n_labels', 5)))
            error_rates = [cer(target_sequences[i], output_sequences[i]) 
                if output_sequences[i] != 'No good transcription' else 0
                for i in range(len(output_sequences))
            ]
            avg_error = sum(error_rates) / len(error_rates)
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
                targets = torch.stack([torch.LongTensor(target) for target in new_targets]).to(device)
                # Compute the loss
                output = output.permute(0, 2, 1)
                loss = initialisation_loss_function(output, targets)
            else:
                # Regular
                # Loss function: CTC
                # Compute the loss
                loss = loss_function(
                    log_softmax(output.permute(1, 0, 2)),
                    target,
                    sequences_lengths,
                    targets_lengths
                )
            # Backward pass
            loss.backward()
            # Gradient step
            optimiser.step()
            if kwargs.get('scheduler', None) == 'OneCycleLR':
                    scheduler.step()
            # Show progress
            train_loss.append(loss.item())
            train_avgcer.append(avg_error)
            if batch_id % 5 == 0:
                print('----------------------------------------------------------------------------------------------------------------------')
                print(f'First target: {target_sequences[0]}\nFirst output: {output_sequences[0]}')
                if kwargs.get('scheduler') is not None:
                    print(f'Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error} Learning rate: {optimiser.param_groups[0]["lr"]}')
                else:
                    print(f'Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error} Learning rate: {optimiser.param_groups[0]["lr"]}')
        # Study test dataset per epoch
        loss, avgcer = test(model, validation_data, loss_function, cer, loss_type='CTCLoss', **kwargs)
        test_loss.append(loss)
        test_avgcer.append(avgcer)
        if kwargs.get('scheduler', None) == 'StepLR':
            scheduler.step()
    # Manual record in file
    record_in_file(train_loss, train_avgcer, test_loss, test_avgcer)
        

def test(model, test_data, loss_function, error_function, loss_type='CTCLoss', **kwargs):
    """
    DESCRIPTION:
    Function to test the model progression during training in the validation
    dataset.
    :param model: [torch.nn.Module] the model to train.
    :param test_data: [iter] the data organised in batches ready to test.
    :param loss_function: [nn.Module] function to test model progression.
    :param error_function: [nn.Module] function to test model progression.
    :param loss_type: [str] label to identify the loss function used.
    """
    model.eval()
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Prepare loss if needed
    if loss_type == 'CTCLoss':
        log_softmax = nn.LogSoftmax(dim=2).to(device)
    # Go through the test batches
    losses = []
    errors = []
    for batch in test_data:
        # Clean gradient
        model.zero_grad()
        # Move data to device
        target_segments = batch['fragments']
        target_sequences = batch['sequences']
        targets_lengths = batch['targets_lengths']
        batch, target = batch['signals'].to(device), batch['targets'].to(device)
        # All the sequences in a batch are of the same length
        sequences_lengths = tuple([batch.shape[-1]] * batch.shape[0])  
        # Forward pass
        output = model(batch)
        # Decode output
        output_sequences = list(decoder(output, kwargs.get('n_labels', 5)))
        error_rates = [error_function(target_sequences[i], output_sequences[i]) 
            if output_sequences[i] != 'No good transcription' else 0
            for i in range(len(output_sequences))
        ]
        avg_error = sum(error_rates) / len(error_rates)
        # Loss
        if loss_type == 'CTCLoss':
            # Loss function: CTC
            # Compute the loss
            loss = loss_function(
                log_softmax(output.permute(1, 0, 2)),
                target,
                sequences_lengths,
                targets_lengths
            )
        else:
            raise KeyError('Invalide loss type declared')
        # Store progress
        losses.append(loss.item())
        errors.append(avg_error)
    model.train()
    # Return the averages
    avg_loss = sum(losses) / len(losses)
    avg_error = sum(errors) / len(errors)
    return avg_loss, avg_error


def train(model, train_data, validation_data, algorithm='single', **kwargs):
    """
    DESCRIPTION:
    Fucntion to abstract the training from the rest of the process.
    :param model: [torch.nn.Module] the model to train.
    :param train_data: [iter] the data organised in batches ready to train.
    :param validation_data: [iter] validation data organised in batches.
    :param algorithm: [str] the type of algorithm to use for batch parallelisation.
    If single, there is no parallelisation. Alternative, DataParallel.
    """
    # Compute training time
    t = TicToc()
    t.tic()
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training started')
    print('Algorithm:', algorithm)
    # Select training algorithm
    if algorithm == 'single':
        model = model.to(device)
        model.train()
        # Start training
        launch_training(model, train_data, validation_data, device, **kwargs)
    elif algorithm == 'DataParallel':
        # Prepare model and data
        model = nn.DataParallel(model)
        model = model.to(device)
        model.train()
        # Start training
        launch_training(model, train_data, validation_data, device, **kwargs)
    else:
        raise ValueError('Invalid training method')
    print('************************************************************')
    t.toc('TRAINING TIME: ')
    print('************************************************************')
