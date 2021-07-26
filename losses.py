#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This files contains custom loss functions designed to
complement the CTC loss.
"""

# Libraries
from metrics import cer
from torch import nn
import torch
from fast_ctc_decode import beam_search


class ETLoss(nn.Module):
    """
    DESCRIPTION:
    Expected Transcription Loss function.
    """
    # Methods
    def __init__(self, metric):
        """
        DESCRIPTION:
        Class constructor.
        :param metric: [def] function to include in the 
        loss computation.
        """
        super().__init__()
        self.metric = metric
    
    def forward(self, probs, targets):
        """
        DESCRIPTION:
        output_sequences = list(decoder(output.view(*output_size)))
        error_rates = [cer(target_sequences[i], output_sequences[i]) for i in range(len(output_sequences))]
        """
        print(probs.shape)
        print(targets.shape)
        batch_size = probs.shape[1]
        output_sequences = list(decoder(probs.view(probs.shape[0], probs.shape[1], -1)))
        error_rates = [cer(targets[i], output_sequences[i]) for i in range(len(output_sequences))]
        print(item_probs.shape)
        print()
        return 3


def decoder(probabilities_matrix, method='greedy'):
    """
    DESCRIPTION:
    The function that implements the greedy algorithm to obtain the
    sequence of letters from the probability matrix.
    :param probabilities_matrix: [torch.Tensor] matrix of dimensions
    [batch_size, sequence_length] with the output probabilities.
    :yield: [str] the sequence associated with a series of 
    probabilities.
    """
    letters = ['A', 'T', 'G', 'C', '$']
    # windows = [length2indices(window) for window in segments_lengths]
    if method == 'greedy':
        max_probabilities = torch.argmax(probabilities_matrix, dim=2)
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
            final_sequence_greedy = final_sequence.replace('$', '')
            prob = probabilities_matrix[i]
            yield final_sequence_greedy
    elif method == 'beam_search':
        probs = probabilities_matrix.cpu().detach().numpy()
        for prob in probs:
            seq, path = beam_search(prob, ''.join(letters), beam_size=5, beam_cut_threshold=1E-24)
            yield seq.replace('$', '')