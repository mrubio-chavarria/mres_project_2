#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
The code below gathers the models developed to test
"""

# Libraries
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import batchnorm
from bnlstm import LSTM
import numpy as np


# Classes
class ResidualBlockI(nn.Module):
    """
    DESCRIPTION:
    Residual block to build the TCN.
    """
    # Methods
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_proportion, optional=False, last=False):
        """
        DESCRIPTION:
        Class constructor.
        :param in_channels: [int] number of input channels of the initial signal.
        :param out_channels: [int] number of filters executed in the convolution.
        :param kernel_size: [int] the number of elements covered by the kernel.
        :param dilation: [int] dilation factor to apply to the whole block.
        :param dropout_proportion: [float] the proportion of neurons to perform the
        dropout.
        :param optional: [bool] an optional parameter to control if there is going to
        be an optional convolution.
        :param last: [bool] an optional parameter to control if the block is the last 
        in the series and, therefore, there should not be a ReLU in the last layer.
        """        
        super().__init__()
        # Optional convolution
        self.optional = optional
        if optional:
            self.optional_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1)
        # Main block section
        if not last:
            self.model = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation
                    )
                ),
                # The first term is the padding size to keep the sequence length
                nn.ConstantPad1d((dilation*(kernel_size - 1), 0), 0),
                nn.ReLU(),
                nn.Dropout(p=dropout_proportion),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation
                    )
                ),
                nn.ConstantPad1d((dilation*(kernel_size - 1), 0), 0),
                nn.ReLU(),
                nn.Dropout(p=dropout_proportion)
            )
        else:
            self.model = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation
                    )
                ),
                # The first term is the padding size to keep the sequence length
                nn.ConstantPad1d((dilation*(kernel_size - 1), 0), 0),
                nn.ReLU(),
                nn.Dropout(p=dropout_proportion),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation
                    )
                ),
                nn.ConstantPad1d((dilation*(kernel_size - 1), 0), 0),
                nn.Dropout(p=dropout_proportion)
            )
    
    def forward(self, input_sequence):
        """
        DESCRIPTION:
        Forward pass of the model class.
        :param input_sequence: [torch.Tensor] the input sequence in which you are
        going to base the prediction.
        :return: [torch.Tensor] the predicted value.
        """
        if self.optional:   
            output = self.model(input_sequence) + self.optional_conv(input_sequence) 
            return output
        else:
            return self.model(input_sequence)

class ResidualBlockII(nn.Module):
    """
    DESCRIPTION:
    Residual block to build the TCN. The design and values of the architecture has
    been directly taken from Chiron.
    """
    # Methods
    def __init__(self, in_channels, out_channels=256, kernel_size=3, dropout=0.5):
        """
        DESCRIPTION:
        Class constructor.
        :param in_channels: [int] number of input channels of the initial signal.
        :param out_channels: [int] number of output channels after filtering.
        :param kernel_size: [int] number of elements in the one-dimensional kernel.
        """        
        super().__init__()
        # Main block section
        self.left_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(num_features=out_channels)
        )
        # Right branch
        self.right_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.final_ReLU = nn.ReLU()

    def forward(self, input_sequence):
        """
        DESCRIPTION:
        Forward pass of the model class.
        :param input_sequence: [torch.Tensor] the input sequence in which you are
        going to base the prediction.
        :return: [torch.Tensor] the predicted value.
        """
        left_branch = self.left_branch(input_sequence)
        right_branch = self.right_branch(input_sequence)
        output = left_branch + right_branch
        output = self.final_ReLU(output)
        return output

class ResidualBlockIII(nn.Module):
    """
    DESCRIPTION:
    FIX VIEW PROBLEM
    Residual block to build the TCN. The architecture has taken from AssemblyAI 
    in https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
    """
    # Methods
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.8):
        """
        DESCRIPTION:
        Class constructor.
        :param in_channels: [int] number of channels (dimensions) in the input
        object.
        :param out_channels: [int] number of channels in the output object.
        """
        super().__init__()
        # First block
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.GELU1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        # Left padding because of the signal causality
        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1)//2)
        # Second block
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.GELU2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1)//2)

    def forward(self, input_sequence):
        """
        DESCRIPTION:
        Forward pass of the model class.
        :param input_sequence: [torch.Tensor] the input sequence in which you are
        going to base the prediction.
        Dimensions: 
        [batch, channels, sequence_length]
        :return: [torch.Tensor] the predicted value.
        """
        dims = input_sequence.shape
        shortcut = input_sequence
        # First block
        # The channels are to be the last dimension
        output = self.layer_norm1(input_sequence.view(dims[0], dims[2], dims[1]))
        output = self.GELU1(output.view(*dims))
        output = self.dropout1(output)
        output = self.cnn1(output)
        dims = output.shape
        # Second block
        # The channels are to be the last dimension
        output = self.layer_norm2(output.view(dims[0], dims[2], dims[1]))
        output = self.GELU2(output.view(*dims))
        output = self.dropout2(output)
        output = self.cnn2(output)
        # Correct shortcut dimension
        if output.shape != shortcut.shape:
            optional_filter = nn.Conv1d(shortcut.shape[1], output.shape[1], kernel_size=1)
            shortcut = optional_filter(shortcut)
        output += shortcut
        return output


class ResidualBlockIV(nn.Module):
    """
    DESCRIPTION:
    Residual block adapted from the model provided by Niccolo McConnel.
    """
    def __init__(self, inchannel, outchannel, stride, kernel_size=3): 

        super().__init__() 
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding_formula(311, stride, kernel_size)

        self.block = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False), 
            nn.BatchNorm1d(outchannel), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(outchannel, outchannel, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False), 
            nn.BatchNorm1d(outchannel)
        ) 

        self.shortcut = nn.Sequential() 

        # takes care of case where input to self.block is different dims than output
        # so that the original input (transformed) can be added again 
        # (i.e. skip connection), note the 1x1 kernel. 
        if inchannel != outchannel: 
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, bias=False), 
                nn.BatchNorm1d(outchannel)
            ) 

    def forward(self, x): 
        # x inputted into block
        out = self.block(x) 
        # original x added with shortcut applied if neccessary to out
        shortcut = self.shortcut(x)
        out += shortcut 
        out = F.relu(out) 
        return out


# Using above defined residual blocks to define ResNet:
class ResNet(nn.Module):

    def __init__(self, ResidualBlock, inchannel):
        super(ResNet, self).__init__()
    
        self.inchannel = 32  # inchannel second layer 
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannel, self.inchannel, kernel_size = 11, stride = 1, padding = 5, bias = False), 
            nn.BatchNorm1d(self.inchannel), 
            nn.ReLU()
        )

        self.layer1 = self.make_layer(ResidualBlock, channels = 64, num_blocks = 2, stride = 2)
        self.layer2 = self.make_layer(ResidualBlock, channels = 128, num_blocks = 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, channels = 256, num_blocks = 2, stride = 2)
        self.layern = self.make_layer(ResidualBlock, channels = 256, num_blocks = 2, stride = 2)

        self.layer4 = self.make_layer(ResidualBlock, channels = 512, num_blocks = 2, stride = 2)
        self.layer5 = self.make_layer(ResidualBlock, channels = 512, num_blocks = 2, stride = 2)
        self.layer6 = self.make_layer(ResidualBlock, channels = 512, num_blocks = 2, stride = 2)
        
        # self.maxpool = nn.MaxPool1d(2)
        # self.dropout_conv = nn.Dropout3d(0.1)  # this is applied on the filter channels 
        # self.fc1 = nn.Linear(512, 1)
        # self.fc2 = nn.Linear(1024, 1)
        # self.drop = nn.Dropout(0.4)  # this is applied on the filter channels 

    # function to create residual layer:
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels # update inchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        # note assuming an input of dim [batch_size, 1,256,256] 
        # since input channel = 1 (2d image)
        x = self.conv1(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        #x = self.layern(x)
        x = self.layer4(x) 
        x = self.layer5(x) 
        x = self.layer5(x) 
        x = self.layer5(x)  # [64, 512, 311]

        # x = self.maxpool(x) # [batch_size, 512, 1, 1] # stop at layer 4: [batch_size, 512, 4, 4]
        # x = x.view(x.size(0), -1) # [batch_size, 512]
        # x = self.fc1(x) # [batch_size, 1]
        # #x = self.drop(x)
        # #x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x.view(64, 512, 311)


class TCN_module(nn.Module):
    """
    DESCRIPTION:
    TCN module to integrate in the final network.
    """
    # Methods
    def __init__(self, n_layers, in_channels, out_channels,  kernel_size=3, dropout=0.8):
        """
        DESCRIPTION:
        Class constructor.
        :param n_layers: [int] number of layers in which the model is built.
        :param in_channels: [int] number of input channels of the initial signal.
        :param out_channels: [list] number of filters executed in the convolution per layer.
        :param dilation_base: [int] base (b) in the formula to compute the dilation
        based on the layer.
        :param kernel_size: [int] the number of elements considered in the kernel.
        :param dropout: [float] the proportion of neurons to perform the dropout.
        :param optional: [bool] parameter to describe the optional convolution in the residual block.
        """
        super().__init__()
        # Parameters to build the residual blocks1
        # Create the model
        blocks = []
        for i in range(n_layers):
            if i == 0:
                blocks.append(
                    ResidualBlockII(in_channels, out_channels, kernel_size)
                )
            else:
                blocks.append(
                    ResidualBlockII(out_channels, out_channels, kernel_size)
                )
        self.model = nn.Sequential(*blocks)
    
    def forward(self, input_sequence):
        """
        DESCRIPTION:
        Forward pass of the model class.
        :param input_sequence: [torch.Tensor] the input sequence in which you are
        going to base the prediction.
        :return: [torch.Tensor] the predicted value.
        """
        return self.model(input_sequence)


class LSTM_module(nn.Module):
    """
    DESCRIPTION:
    LSTM module to integrate in the final network.
    """
    # Methods
    def __init__(self, n_layers, input_size, batch_size, hidden_size, dropout=0.2, bidirectional=False, batch_first=True):
        """
        DESCRIPTION:
        Class constructor.
        :param n_layers: [int] number of LSTM to build.
        :param sequence_length: [int] number of elements in the input tensor.
        :param input_size: [int] dimensionality of the input space.
        :param batch_size: [int] number of items per batch.
        :param hidden_size: [int] features in the hidden space.
        :param dropout: [float] proportion of dropout neurons.
        :param bidirectional: [bool] variable to indicate if the LSTM layers
        are bidirectional. 
        """
        super().__init__()
        # Parameters
        self.n_layers = n_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout_proportion = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.hidden_cell_state = (torch.zeros(1, batch_size, hidden_size),
                                  torch.zeros(1, batch_size, hidden_size))
        # LSTM layers
        # # Pytorch's LSTM
        # # Pytorch LSTM module to compare if needed
        # self.model = nn.LSTM(input_size, hidden_size, num_layers=n_layers,
        #     batch_first=batch_first, bidirectional=bidirectional)
        # BatchNorm LSTM
        self.model = LSTM(input_size, hidden_size, n_layers, batch_first=batch_first,
            bidirectional=bidirectional, batch_norm=True)

    
    def forward(self, input_sequence):
        """
        DESCRIPTION:
        Forward pass of the model class.
        :param input_sequence: [torch.Tensor] input sequence. Dimensions:
        [batch_size, channels, sequence_length]
        :return: [float] predicted value.
        """
        # We do not store the hidden and cell states
        # When bidirectional, the output dim is 2 * hidden dim
        # # Pytorch's LSTM
        # # Pytorch LSTM module to compare if needed
        # if self.batch_first:
        #     output, _ = self.model(input_sequence.permute(0, 2, 1))
        # else:
        #     output, _ = self.model(input_sequence.permute(2, 0, 1))
        # Batch Norm LSTM
        if self.batch_first:
            output, _ = self.model(input_sequence.permute(0, 2, 1))
        else:
            output, _ = self.model(input_sequence.permute(2, 0, 1))
        return output


class DecoderGELU(nn.Module):
    """
    DESCRIPTION:
    Model to assign the probabilities of every base for a given signal.
    """
    # Methods
    def __init__(self, initial_size, output_size, sequence_length, batch_size, dropout=0.2):
        """
        DESCRIPTION:
        Class constructor.
        Important, the softmax is already implemented in the cost function.
        :param initial_size: [int] input dimensionality.
        :param output_size: [int] output dimensionality. Number of
        classes.
        :param sequence_length: [int] number of elements in the input sequence.
        :param batch_size: [int] number of elements per batch.
        :param dropout: [float] proportion of dropout neurons.
        """
        super().__init__()
        self.initial_size = initial_size
        self.hidden_size = 2 * initial_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.Linear(self.initial_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, input_sequence):
        """
        DDESCRIPTION:
        Forward pass.
        :param input_sequence: [torch.Tensor] the sequence to feed the model.
        """
        output = self.model(input_sequence)
        return output


class DecoderChiron(nn.Module):
    """
    DESCRIPTION:
    Model to assign the probabilities of every base for a given signal.
    """
    # Methods
    def __init__(self, initial_size, output_size, batch_size, dropout=0.8):
        """
        DESCRIPTION:
        Class constructor.
        Important, the softmax is already implemented in the cost function.
        :param initial_size: [int] input dimensionality.
        :param output_size: [int] output dimensionality. Number of
        classes.
        :param batch_size: [int] number of elements per batch.
        :param dropout: [float] proportion of dropout neurons.
        """
        super().__init__()
        self.initial_size = initial_size
        self.hidden_size = 2 * initial_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.model = nn.Sequential(
            nn.Linear(self.initial_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, input_sequence):
        """
        DDESCRIPTION:
        Forward pass.
        :param input_sequence: [torch.Tensor] the sequence to feed the model.
        """
        output = self.model(input_sequence)
        return output


class DecoderCustom(nn.Module):
    """
    DESCRIPTION:
    Model to assign the probabilities of every base for a given signal.
    """
    # Methods
    def __init__(self, initial_size, output_size, batch_size, dropout=0.8):
        """
        DESCRIPTION:
        Class constructor.
        Important, the softmax is already implemented in the cost function.
        :param initial_size: [int] input dimensionality.
        :param output_size: [int] output dimensionality. Number of
        classes.
        :param batch_size: [int] number of elements per batch.
        :param dropout: [float] proportion of dropout neurons.
        """
        super().__init__()
        self.initial_size = initial_size
        self.hidden_size = 2 * initial_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Linear(self.initial_size, self.hidden_size),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
    def forward(self, input_sequence):
        """
        DDESCRIPTION:
        Forward pass.
        :param input_sequence: [torch.Tensor] the sequence to feed the model.
        """
        # Prepare the constant weights and bias
        output = self.model(input_sequence)
        return output


# Functions
def dilation_formula(layer_index, dilation_base):
    """
    DESCRIPTION:
    A formula that tells you the dilation based on the base and the layer
    index.
    :param dilation_base: [int] the base to compute the dilation, when no 
    dilation, base = 1.
    :param layer_index: [int] the level of the layer.
    """
    return (dilation_base ** layer_index)


def padding_formula(sequence_length, stride, kernel_size):
    return int(round(0.5 * ((stride - 1) * sequence_length + kernel_size - stride)))