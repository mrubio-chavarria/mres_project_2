#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The functions in this file develop the training in the 
experiment scripts.
"""

# Libraries
from typing_extensions import final
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from datasets import collate_text2int_fn
from metrics import cer
from torch import multiprocessing as mp


# Functions
def length2indices(window):
    indeces = [0]
    for i in range(len(window)):
        indeces.append(window[i] + indeces[i])
    return indeces


def decoder(probabilities_matrix):
    """
    DESCRIPTION:
    The function that implements the greedy algorithm to obtain the
    sequence of letters from the probability matrix.
    :param probabilities_matrix: [torch.Tensor] matrix of dimensions
    [batch_size, sequence_length] with the output probabilities.
    :yield: [str] the sequence associated with a series of 
    probabilities.
    """
    max_probabilities = torch.argmax(probabilities_matrix, dim=2)
    letters = ['A', 'T', 'G', 'C', '$']
    # windows = [length2indices(window) for window in segments_lengths]
    for i in range(len(probabilities_matrix)):
        # Output probabilities to sequence
        # OLD
        # sequence = [letters[prob] for prob in max_probabilities[i].tolist()]
        # sequence = [sequence[windows[i][index]:windows[i][index+1]] for index in range(len(windows[i])-1)]
        # sequence = [''.join(list(set(segment))) for segment in sequence]
        # sequence = ''.join(sequence)
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
        yield final_sequence.replace('$', '')


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


def launch_training(model, train_data, device, experiment, rank=0, sampler=None, **kwargs):
    """
    DESCRIPTION:
    UPDATE
    A function to launch the model training.
    :param rank: [int] index of the process executing the function.
    :param sampler: [DistributedSampler] sampler to control batch reordering
    after eery epoch in the case of multiprocessing.
    COMPLETE
    """
    # Create optimiser
    if kwargs.get('optimiser', 'SGD') == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-4),
                                    momentum=kwargs.get('momemtum', 0))
    elif kwargs.get('optimiser', 'SGD') == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(),
                                    lr=kwargs.get('learning_rate', 1E-4),
                                    weight_decay=kwargs.get('weight_decay', 0))
    elif kwargs.get('optimiser') == 'RMSprop':
        optimiser = torch.optim.RMSprop(model.parameters(),
                                        lr=kwargs.get('learning_rate', 1E-3),
                                        weight_decay=kwargs.get('weight_decay',0),
                                        momentum=kwargs.get('momemtum', 0))
    else:
        raise ValueError('Invalid optimiser selected')
    # Max number of batches
    max_batches = kwargs.get('max_batches', 500)
    # Create scheduler
    if kwargs.get('scheduler') is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
                                                        max_lr=kwargs.get('max_learning_rate', 1E-2),
                                                        steps_per_epoch=len(train_data),
                                                        epochs=kwargs.get('n_epochs', 30))
    # Prepare training
    batch_size = kwargs.get('batch_size')
    # sequences_lengths = tuple([sequence_length] * batch_size)
    log_softmax = nn.LogSoftmax(dim=2).to(device)
    loss_function = nn.CTCLoss(blank=4).to(device)
    initialisation_loss_function = nn.CrossEntropyLoss().to(device)
    initialisation_epochs = range(kwargs.get('n_initialisation_epochs', 1))
    # Train
    with experiment.train():
        for epoch in range(kwargs.get('n_epochs', 1)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch_id, batch in enumerate(train_data):
                if batch_id == max_batches:
                    break
                # Clean gradient
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
                # Loss
                # Set different loss to initialise
                output_size = output.shape
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
                output_sequences = list(decoder(output.view(*output_size)))
                error_rates = [cer(target_sequences[i], output_sequences[i]) for i in range(len(output_sequences))]
                avg_error = sum(error_rates) / len(error_rates)
                # Show progress
                if batch_id % 25 == 0:
                    print('----------------------------------------------------------------------------------------------------------------------')
                    print(f'First target: {target_sequences[0]}\nFirst output: {output_sequences[0]}')
                    # print(f'First target: {target_sequences[0]}\nFirst output: {seq}')
                    if kwargs.get('scheduler') is not None:
                        print(f'Process: {rank} Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error} Learning rate: {optimiser.param_groups[0]["lr"]}')
                    else:
                        print(f'Process: {rank} Epoch: {epoch} Batch: {batch_id} Loss: {loss} Error: {avg_error}')
            
                # Record data by batch
                experiment.log_metric('loss', loss.item(), step=batch_id, epoch=epoch)
                experiment.log_metric('learning_rate', optimiser.param_groups[0]["lr"], step=batch_id, epoch=epoch)
                experiment.log_metric('avg_batch_error', avg_error, step=batch_id, epoch=epoch)
            # Record data by epoch
            experiment.log_metric('loss', loss.item(), epoch=epoch)
            experiment.log_metric('learning_rate', optimiser.param_groups[0]["lr"], epoch=epoch)
            experiment.log_metric('avg_batch_error', avg_error, step=batch_id, epoch=epoch)


def train(model, train_data, experiment, algorithm='single', n_processes=3, **kwargs):
    """
    DESCRIPTION:
    UPDATE
    A wrapper to control the use of multiprocessing Hogwild algorithm from single
    node traditional training.
    :param algorithm: [str] the algorithm to train. Classic single-node and CPU 
    trainin, or single-node multi CPU Hogwild.
    :param model: [torch.nn.Module] the model to train.
    COMPLETE
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training started')
    print('Algorithm:', algorithm)
    # Select training algorithm
    if algorithm == 'single':
        model = model.to(device)
        model.train()
        # Start training
        launch_training(model, train_data, device, experiment, **kwargs)
    elif algorithm == 'DataParallel':
        # Prepare model and data
        model = nn.DataParallel(model)
        model = model.to(device)
        model.train()
        # Start training
        launch_training(model, train_data, device, experiment, **kwargs)
    # elif algorithm == 'Hogwild':
    #     # Start training
    #     # We are not setting a blockade per epoch
    #     model = model.to(device)
    #     model.train()
    #     model.share_memory()
    #     processes = []
    #     for rank in range(n_processes):
    #         train_sampler = DistributedSampler(train_dataset, num_replicas=n_processes, rank=rank)
    #         train_data = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=kwargs.get('batch_size'), collate_fn=collate_text2int_fn)
            
    #         print(f'Process {rank} launched')
    #         process = mp.Process(target=launch_training, args=(model, train_data, device, experiment, rank, train_sampler), kwargs=kwargs)
    #         process.start()
    #         processes.append(process)
    #     for process in processes:
    #         process.join()
    else:
        raise ValueError('Invalid training method')
