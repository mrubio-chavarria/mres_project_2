#!/home/mario/anaconda3/envs/project2_venv/bin python

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import batchnorm

# Classes
class LSTMlayer(nn.Module):
    """
    DESCRIPTION:
    The class to stack the LSTM cells in a single layer.
    """
    # Methods
    def __init__(self, input_size, hidden_size, batch_size, layer_index, reference=None, method='uniform', bidirectional=False, batch_norm=False):
        """
        DESCRIPTION:
        Class constructor.
        :param input_size: [int] dimensionality of every item in the input sequence.
        :param hidden_size: [int] dimensionality of every sequence item in the hidden space.
        :param batch_size: [int] number of sequences per batch.
        :param layer_index: [int] a number that indicates the depth of the layer.
        :param reference: [torch.nn.LSTM] a Pytorch LSTM to import the weights from.
        :param method: [str] the method to initialise weight matrices.
        :param bidirectional: [bool] flag to indicate if the layers building the model should
        be bidirectional or not.
        :param batch_norm: [bool] flag to indicate if there should be recurrent batch
        normalisation.
        """
        # Helper function
        def initialise_matrix(dims, method='uniform'):
            """
            DESCRIPTION:
            Function to initialise matrices based on the uniform or othogonal methods.
            :param dims: [list] dimensions of the matrix to initialise.
            :param params: [list] list with the parameters to initialise the matrices.
            :param method: [str] the method to initialise.
            :return: [torch.Tensor] the initialised matrix.
            """
            uniform_params = [(1 / hidden_size) ** 0.5] * 2
            orthogonal_params = [1]  # gain
            if len(dims) == 2:
                matrices = [torch.empty(int(dims[0] / 4), dims[1]) for _ in range(4)]
            else:
                matrices = [torch.empty(int(dims[0] / 4)) for _ in range(4)]
            if method == 'uniform':
                matrices = [init.uniform_(matrix, *uniform_params) for matrix in matrices]
            elif method == 'orthogonal':
                if len(dims) == 2:
                    matrices = [init.orthogonal_(matrix, *orthogonal_params) for matrix in matrices]
                else:
                    # Vector matrices (biases) cannot be orthogonal
                    matrices = [init.uniform_(matrix, *uniform_params) for matrix in matrices]
            else:
                raise ValueError('Incorrect matrix initialisation method chosen.')
            return torch.cat(matrices, dim=0)

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layer_index = layer_index
        self.bidirectional = bidirectional
        self.weigths_initialisation_method = method
        # Create the batch norms if needed
        self.batch_norms = None
        self.batch_norms_reverse = None
        if batch_norm:
            # Define the batch normalisations to use
            # After the matrix multiplication, there is only 1 dimension
            # c_t is going to have hidden size 
            self.batch_norms = [
                # weight_hh @ h_t_1
                nn.BatchNorm1d(1),
                # weight_hh @ x_t 
                nn.BatchNorm1d(1),
                # c_t_1
                nn.BatchNorm1d(hidden_size)
            ]
            if bidirectional:
                self.batch_norms_reverse = [
                    # weight_hh @ h_t_1
                    nn.BatchNorm1d(1),
                    # weight_hh @ x_t 
                    nn.BatchNorm1d(1),
                    # c_t_1
                    nn.BatchNorm1d(hidden_size)
                ]
            # Define the cell
            self.cell = bnlstm_cell
        else:
            # Define the cell
            self.cell = lstm_cell
        # Import or create the matrices
        if reference is None:
            # Create
            dims = [4 * hidden_size, input_size]
            self.weight_ih = initialise_matrix(dims, method)
            dims = [4 * hidden_size, hidden_size]
            self.weight_hh = initialise_matrix(dims, method)
            dims = [4 * hidden_size]
            self.bias_ih = initialise_matrix(dims, method)
            self.bias_hh = initialise_matrix(dims, method)
            if self.bidirectional:
                dims = [4 * hidden_size, input_size]
                self.weight_ih_reverse = initialise_matrix(dims, method)
                dims = [4 * hidden_size, hidden_size]
                self.weight_hh_reverse = initialise_matrix(dims, method)
                dims = [4 * hidden_size]
                self.bias_ih_reverse = initialise_matrix(dims, method)
                self.bias_hh_reverse = initialise_matrix(dims, method)
        else:
            # Import
            setattr(self, f'weight_ih', getattr(reference, f'weight_ih_l{layer_index}'))
            setattr(self, f'weight_hh', getattr(reference, f'weight_hh_l{layer_index}'))
            setattr(self, f'bias_ih', getattr(reference, f'bias_ih_l{layer_index}'))
            setattr(self, f'bias_hh', getattr(reference, f'bias_hh_l{layer_index}'))
            if self.bidirectional:
                setattr(self, f'weight_ih_reverse', getattr(reference, f'weight_ih_l{layer_index}_reverse'))
                setattr(self, f'weight_hh_reverse', getattr(reference, f'weight_hh_l{layer_index}_reverse'))
                setattr(self, f'bias_ih_reverse', getattr(reference, f'bias_ih_l{layer_index}_reverse'))
                setattr(self, f'bias_hh_reverse', getattr(reference, f'bias_hh_l{layer_index}_reverse'))
        # Send weights to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device  # Store for future calculations
        self.weight_ih = self.weight_ih.to(device)
        self.weight_hh = self.weight_hh.to(device)
        self.bias_ih = self.bias_ih.to(device)
        self.bias_hh = self.bias_hh.to(device)
        if self.bidirectional:
            self.weight_ih_reverse = self.weight_ih_reverse.to(device)
            self.weight_hh_reverse = self.weight_hh_reverse.to(device)
            self.bias_ih_reverse = self.bias_ih_reverse.to(device)
            self.bias_hh_reverse = self.bias_hh_reverse.to(device)

    def forward(self, sequence, initial_states=None):
        """
        DESCRIPTION:
        Forward pass.
        :param sequence: [torch.Tensor] sequence to feed the network. Dimensionality:
        [sequence_length, batch_size, input_size].
        :param initial_states: [tuple] (h_0, c_0), the initial hidden and cell states.
        Both with the dimensionality: [directions * number_of_layers, batch_size, hidden_size].
        If initial_states == None, they are created with zeros downwards.
        :return: [torch.Tensor] the hidden state of the whole layer [sequence_length,
        batch_size, n_directions*hidden_size]. And the last hidden and cell states with
        dimensionality: [n_directions, batch_size, hidden_size] (the code follows when possible
        the Pytorch convention, with num_layers==1).
        """
        # Helper functions
        def sequence_pass(sequence, h_0, c_0, reverse=False):
            """
            DESCRIPTION:
            Helper function to go through the sequence in one way or the other. 
            :param sequence: [torch.Tensor] sequence to feed the network. Dimensionality:
            [sequence_length, batch_size, input_size].
            :param initial_states: [tuple] (h_0, c_0), the initial hidden and cell states.
            Both with the dimensionality: [directions * number_of_layers, batch_size, hidden_size].
            :param reverse: [bool] a flag to indicate if the sequence should computed in 
            reversed order.
            :return: [torch.Tensor] the hidden and cell spaces of the whole layer without h_0 and
            c_0. Both with dimensionality: [sequence_length, batch_size, hidden_size].
            """
            hs = [h_0]
            cs = [c_0]
            if reverse:
                for i in range(sequence.shape[0]-1, -1, -1):
                    h_t, c_t = self.cell(sequence[i].view(self.batch_size, self.input_size, 1),
                                        hs[-1].view(self.batch_size, self.hidden_size, 1), 
                                        cs[-1].view(self.batch_size, self.hidden_size, 1), 
                                        self.weight_ih_reverse, self.weight_hh_reverse, self.bias_ih_reverse, self.bias_hh_reverse, self.batch_norms_reverse)
                    hs.append(h_t)
                    cs.append(c_t)
                h = torch.flip(torch.stack(hs[1:], dim=0), dims=(0,)).view(sequence.shape[0], -1, self.hidden_size)
                c = torch.flip(torch.stack(cs[1:], dim=0), dims=(0,)).view(sequence.shape[0], -1, self.hidden_size)
            else:
                for i in range(sequence.shape[0]):
                    h_t, c_t = self.cell(sequence[i].view(self.batch_size, self.input_size, 1), 
                                        hs[-1].view(self.batch_size, self.hidden_size, 1), 
                                        cs[-1].view(self.batch_size, self.hidden_size, 1),
                                        self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.batch_norms)
                    hs.append(h_t)
                    cs.append(c_t)
                h = torch.stack(hs[1:], dim=0).view(sequence.shape[0], -1, self.hidden_size)
                c = torch.stack(cs[1:], dim=0).view(sequence.shape[0], -1, self.hidden_size)
            return h, c

        # Initialise hidden and cell states
        if initial_states is None:
            h_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.batch_size, self.hidden_size).to(self.device)
        else:
            h_0, c_0 = initial_states
        # Forward pass
        h_f, c_f = sequence_pass(sequence, h_0, c_0, reverse=False)
        if self.bidirectional:
            # Backward pass
            h_b, c_b = sequence_pass(sequence, h_0, c_0, reverse=True)
            # Concat both
            h = torch.cat((h_f, h_b), dim=2)
            c = torch.cat((c_f, c_b), dim=2)
        else:
            h, c = h_f, c_f
        n_directions = 2 if self.bidirectional else 1
        h_n = h[-1,:].view(n_directions, self.batch_size, self.hidden_size)
        c_n = c[-1,:].view(n_directions, self.batch_size, self.hidden_size)
        return h, (h_n, c_n)


class LSTM(nn.Module):
    """
    DESCRIPTION:
    Final LSTM module built on the previous ones.
    """
    # Methods
    def __init__(self, input_size, hidden_size, num_layers, batch_size, batch_first=False, reference=None, method='uniform', bidirectional=False, batch_norm=False):
        """
        Class constructor.
        :param input_size: [int] dimensionality of every item in the input sequence.
        :param hidden_size: [int] dimensionality of every sequence item in the hidden space.
        :param batch_size: [int] number of sequences per batch.
        :param batch_first: [bool] if True, the input should be
        [batch_size, sequence_length, n_features], otherwise 
        [sequence_length, batch_size, n_features]
        :param num_layers: [int] number of layers that compound the model.
        :param reference: [torch.nn.LSTM] a model to import the weights from.
        :param method: [str] the method to initialise the matrices.
        :param bidirectional: [bool] flag to indicate if the layers building the model should
        be bidirectional or not.
        :param batch_norm: [bool] flag to indicate if there should be recurrent batch
        normalisation.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.n_directions = 2 if bidirectional else 1
        # Create the layers
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(LSTMlayer(input_size, hidden_size, batch_size, i, reference, method, bidirectional, batch_norm))
            else:
                self.layers.append(LSTMlayer(self.n_directions * hidden_size, hidden_size, batch_size, i, reference, method, bidirectional, batch_norm))
    
    def forward(self, sequence):
        """
        DESCRIPTION:
        Forward pass.
        :param sequence: [torch.Tensor] sequence to feed the network. Dimensionality:
        [sequence_length, batch_size, input_size].
        :param initial_states: [tuple] (h_0, c_0), the initial hidden and cell states.
        Both with the dimensionality: [directions * number_of_layers, batch_size, hidden_size].
        If initial_states == None, they will be created with zeros downwards.
        :return: the of torch.nn.LSTM. [torch.Tensor] the hidden state of the last layer,
        dimensionality: [sequence_length, batch_size, number_of_directions*hidden_size]. The 
        final tuple are the last-neuron hidden and cell states, both of dimensionality:
        [number_of_directions*hidden_size, batch_size, hidden_size].
        """
        # Correct for the batch_first
        if self.batch_first:
            sequence = sequence.view(-1, self.batch_size, self.input_size)
        h_n = []
        c_n = []
        output = sequence
        for layer in self.layers:
            output, (h_t, c_t) = layer(output)
            h_n.append(h_t)
            c_n.append(c_t)
        h_n = torch.stack(h_n, dim=0).view(-1, self.batch_size, self.hidden_size)
        c_n = torch.stack(c_n, dim=0).view(-1, self.batch_size, self.hidden_size)
        return output, (h_n, c_n)    


# Functions
def lstm_cell(x, h_t_1, c_t_1, weight_ih, weight_hh, bias_ih, bias_hh, *args):
    """
    DESCRIPTION:
    Function that represents the basic LSTM cell. A function to iterate over 
    the sequence.
    :param x: [torch.Tensor] cell input, value in the sequence for a time step.
    Dimensionality: [batch_size, input_size, 1]. The second is 1 because it is
    always one item from the general sequence.
    :param h_t_1: [torch.Tensor] hidden state of the previous cell.
    Dimensionality: [batch_size, hidden_size, 1].
    :param c_t_1: [torch.Tensor] cell state of the previous cell.
    Dimensionality: [batch_size, hidden_size, 1].
    :param weight_ih: [torch.Tensor] layer weights to multiply by the input.
    :param weight_hh: [torch.Tensor] layer weights t multiply by the previous
    hidden state.
    :param bias_ih: [torch.Tensor] layer bias of the input.
    :param bias_hh: [torch.Tensor] layer bias of the output.
    :return: hidden and cell states associated with this time step (h_t, c_t),
    both with dimensionality: [batch_size, hidden_size, 1].
    """
    weight_hh = torch.stack([weight_hh] * h_t_1.shape[0], dim=0)
    weight_ih = torch.stack([weight_ih] * x.shape[0], dim=0)
    bias_hh = torch.stack([bias_hh] * x.shape[0], dim=0).view(x.shape[0], -1, 1)
    bias_ih = torch.stack([bias_ih] * x.shape[0], dim=0).view(x.shape[0], -1, 1)
    ifgo = weight_hh @ h_t_1 + bias_hh + weight_ih @ x + bias_ih
    i, f, g, o = torch.split(ifgo, int(weight_ih.shape[1] / 4), dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c_t = f * c_t_1 + i * g
    h_t = o * torch.tanh(c_t)
    return h_t, c_t


def bnlstm_cell(x, h_t_1, c_t_1, weight_ih, weight_hh, bias_ih, bias_hh, batch_norms):
    """
    DESCRIPTION:
    Function that represents the LSTM cell with batch normalisation. Adapted
    from: arXiv:1603.09025 [cs.LG]
    :param x: [torch.Tensor] cell input, value in the sequence for a time step.
    Dimensionality: [batch_size, input_size, 1]. The second is 1 because it is
    always one item from the general sequence.
    :param h_t_1: [torch.Tensor] hidden state of the previous cell.
    Dimensionality: [batch_size, hidden_size, 1].
    :param c_t_1: [torch.Tensor] cell state of the previous cell.
    Dimensionality: [batch_size, hidden_size, 1].
    :param weight_ih: [torch.Tensor] layer weights to multiply by the input.
    :param weight_hh: [torch.Tensor] layer weights t multiply by the previous
    hidden state.
    :param bias_ih: [torch.Tensor] layer bias of the input.
    :param bias_hh: [torch.Tensor] layer bias of the output.
    :param batch_norms: [list] vector to store the torch.BatchNorm1d that 
    we will apply, one per transformation and direction.
    :return: hidden and cell states associated with this time step (h_t, c_t),
    both with dimensionality: [batch_size, hidden_size, 1].
    """
    weight_hh = torch.stack([weight_hh] * h_t_1.shape[0], dim=0)
    weight_ih = torch.stack([weight_ih] * x.shape[0], dim=0)
    bias_hh = torch.stack([bias_hh] * x.shape[0], dim=0).view(x.shape[0], -1, 1)
    bias_ih = torch.stack([bias_ih] * x.shape[0], dim=0).view(x.shape[0], -1, 1)
    # Apply the first batch normalisations
    weight_hh_by_h_t_1 = weight_hh @ h_t_1
    dims = weight_hh_by_h_t_1.shape
    weight_hh_by_h_t_1 = weight_hh_by_h_t_1.view(dims[0], dims[2], dims[1])
    weight_hh_by_h_t_1 = batch_norms[0](weight_hh_by_h_t_1).view(*dims)
    weight_ih_by_x = weight_ih @ x
    weight_ih_by_x = batch_norms[1](weight_ih_by_x.view(dims[0], dims[2], dims[1])).view(*dims)
    ifgo = weight_hh_by_h_t_1 + bias_hh + weight_ih_by_x + bias_ih
    i, f, g, o = torch.split(ifgo, int(weight_ih.shape[1] / 4), dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c_t = f * c_t_1 + i * g
    # Apply the last batch normalisation
    h_t = o * torch.tanh(batch_norms[2](c_t))
    return h_t, c_t


    

