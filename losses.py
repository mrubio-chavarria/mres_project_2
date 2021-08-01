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
        # DEVELOPMENT
        return 3


class FocalCTCLoss(nn.Module):
    """
    DESCRIPTION:
    Implementation of the focal CTC loss introduced in:
    https://doi.org/10.1155/2019/9345861
    """
    # Methods
    def __init__(self, alpha=0.25, gamma=0.5, blank=0):
        """
        DESCRIPTION:
        Class constructor.
        :param alpha: [float] first focal hyperparameter.
        :param gamma: [float] second focal hyperparameter.
        :param blank: [int] position of the blank character in the input probabilities.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.blank = blank
        self.ctc_loss = nn.CTCLoss(blank=blank)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        DESCRIPTION:
        Forward pass.
        :param log_probs: [torch.Tensor] the log softmax input probabilities. 
        outputted by the network. Dimensionality: [sequence_length, batch_size, n_classes].
        :param targets: [torch.Tensor] ints containing the label for every time step in the
        sequences. All the samples in the batch are concatenated.
        :param input_lengths: [tuple] ints describing the size of every input in the batch.
        :param target_lengths: [tuple] ints describing the length of the label of every 
        sample in the batch.
        :return: [torch.Tensor] loss value.
        """
        ctc_value = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return self.alpha * (1 - torch.exp(-ctc_value)) ** self.gamma * ctc_value

