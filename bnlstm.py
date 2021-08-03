#!/home/mario/anaconda3/envs/project2_venv/bin python

import torch
from torch import device, nn
from torch.nn import init

# Classes
class BatchNormModule(nn.Module):
    """
    DESCRIPTION:
    Module to store the batch normalisation (BN) equations to apply per timestep and layer.
    The whole class has been adapted from:
    https://github.com/jihunchoi/recurrent-batch-normalization-pytorch
    """
    # Methods
    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1, affine=True, zero_bias=True):
        """
        DESCRIPTION:
        Class constructor.
        :param num_features: [int] number of features over which the BN should be computed.
        :param max_length: [int] maximum number of positions to register in the buffer. It 
        correspond to the maximum sequence length to be computed. 
        :param eps: [float] number to avoid division by 0 in BN formula.
        :param momentum: [float] value to assign in Pytorch's BN formula downwards. 
        :param affine: [bool] parameter to set the parameters as learnable.
        :param zero_bias: [bool] flag to indicate whether the bias should be initialised as
        zero or not, default True.
        """
        super().__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.zero_bias = zero_bias 
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))#.to(self.device)
            self.bias = nn.Parameter(torch.FloatTensor(num_features))#.to(self.device)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                f'running_mean_{i}', torch.zeros(num_features)#.to(self.device)
            )
            self.register_buffer(
                f'running_var_{i}', torch.zeros(num_features)#.to(self.device)
            )

        self.reset_parameters()

    def __repr__(self) -> str:
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def reset_parameters(self):
        """
        DESCRIPTION:
        Function to initialise the running mean and variances values to 0 and 1
        respectively. Similarly, the weights are randomly initialised for the 
        case of the linear weights, the biases are set to 0.
        """
        for i in range(self.max_length):
            running_mean_i = getattr(self, f'running_mean_{i}')
            running_var_i = getattr(self, f'running_var_{i}')
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_() if self.zero_bias else self.bias.data.uniform_()

    def _check_input_dim(self, input_sequence):
        """
        DESCRIPTION:
        Value to check that the number of dimensions in the input coincides with
        the dimensions in the running means and variances.
        :param inpute_sequence: [torch.Tensor] input tensor. Dimensionality: 
        [batch_size, input_size] The input size of this function depending on
        the instatiation, not the network input size.
        """
        if input_sequence.size(1) != self.running_mean_0.nelement():
            raise ValueError(
                f'got {input_sequence.size(1)}-feature tensor, expected {self.num_features}'
            )

    def forward(self, input_sequence, time):
        """
        DESCRIPTION:
        Module forward pass.
        :param inpute_sequence: [torch.Tensor] input tensor. Dimensionality: 
        [batch_size, input_size] The input size of this function depending on
        the instatiation, not the network input size.
        """
        # Check that the dimensions coincide
        self._check_input_dim(input_sequence)
        # If the time exceeds the limit, set those values in the last slot8
        if time >= self.max_length:
            time = self.max_length - 1
        # Get the running statistics
        running_mean = getattr(self, f'running_mean_{time}')
        running_var = getattr(self, f'running_var_{time}')
        # Compute values
        output = nn.functional.batch_norm(
            input=input_sequence, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)
        return output

class LSTMlayer(nn.Module):
    """
    DESCRIPTION:
    The class to stack the LSTM cells in a single layer.
    """
    # Methods
    def __init__(self, input_size, hidden_size, layer_index, reference=None, bidirectional=False, batch_norm=False, gamma=0.1, max_length=1000):
        """
        DESCRIPTION:
        Class constructor.
        :param input_size: [int] dimensionality of every item in the input sequence.
        :param hidden_size: [int] dimensionality of every sequence item in the hidden space.
        :param layer_index: [int] a number that indicates the depth of the layer.
        :param reference: [torch.nn.LSTM] a Pytorch LSTM to import the weights from.
        :param bidirectional: [bool] flag to indicate if the layers building the model should
        be bidirectional or not.
        :param batch_norm: [bool] flag to indicate if there should be recurrent batch
        normalisation.
        :param gamma: [float] gamma value to initialse the batch norms.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_index = layer_index
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        # Define the cell
        self.cell = bnlstm_cell if batch_norm else lstm_cell
        if batch_norm:
            # Batch normalisation parameters
            self.bn_ih = BatchNormModule(4 * hidden_size, max_length=max_length, zero_bias=True)
            self.bn_hh = BatchNormModule(4 * hidden_size, max_length=max_length, zero_bias=True)
            self.bn_c = BatchNormModule(hidden_size, max_length=max_length, zero_bias=True)
            # Initialise the parameters
            self.bn_ih.reset_parameters()
            self.bn_hh.reset_parameters()
            self.bn_c.reset_parameters()
            self.bn_ih.bias.data.fill_(0)
            self.bn_hh.bias.data.fill_(0)
            self.bn_ih.weight.data.fill_(gamma)
            self.bn_hh.weight.data.fill_(gamma)
            self.bn_c.weight.data.fill_(gamma)

        # Import or create the matrices
        if reference is None:
            # Create
            dims = [4 * hidden_size, input_size]
            self.weight_ih = self.initialise_matrix(dims, method='orthogonal')
            dims = [4 * hidden_size, hidden_size]
            self.weight_hh = self.initialise_matrix(dims, method='identity')
            dims = [4 * hidden_size]
            self.bias = self.initialise_matrix(dims, method='zeros')
            if self.bidirectional:
                dims = [4 * hidden_size, input_size]
                self.weight_ih_reverse = self.initialise_matrix(dims, method='orthogonal')
                dims = [4 * hidden_size, hidden_size]
                self.weight_hh_reverse = self.initialise_matrix(dims, method='identity')
                dims = [4 * hidden_size]
                self.bias_reverse = self.initialise_matrix(dims, method='zeros')
        else:
            # Import
            setattr(self, f'weight_ih', getattr(reference, f'weight_ih_l{layer_index}'))
            setattr(self, f'weight_hh', getattr(reference, f'weight_hh_l{layer_index}'))
            setattr(self, f'bias', getattr(reference, f'bias_ih_l{layer_index}') + getattr(reference, f'bias_hh_l{layer_index}'))
            if self.bidirectional:
                setattr(self, f'weight_ih_reverse', getattr(reference, f'weight_ih_l{layer_index}_reverse'))
                setattr(self, f'weight_hh_reverse', getattr(reference, f'weight_hh_l{layer_index}_reverse'))
                setattr(self, f'bias_reverse', getattr(reference, f'bias_ih_l{layer_index}_reverse') + getattr(reference, f'bias_hh_l{layer_index}_reverse'))
        # Send weights to device and format biases dimensions
        self.weight_ih = self.weight_ih.cuda
        self.weight_hh = self.weight_hh
        self.bias = nn.Parameter(torch.unsqueeze(self.bias, 1))
        if self.bidirectional:
            self.weight_ih_reverse = self.weight_ih_reverse
            self.weight_hh_reverse = self.weight_hh_reverse
            self.bias_reverse = nn.Parameter(self.bias_reverse.view(-1, 1))

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

        # Initialise hidden and cell states
        if initial_states is None:
            h_0 = torch.zeros(self.hidden_size, sequence.shape[1]).cuda()
            c_0 = torch.zeros(self.hidden_size, sequence.shape[1]).cuda()
        else:
            h_0, c_0 = initial_states
        h_0 = h_0
        c_0 = c_0
        # Forward pass
        h_f, c_f = self.sequence_pass(sequence, h_0, c_0, reverse=False)
        h_f = h_f
        c_f = c_f
        if self.bidirectional:
            # Backward pass
            h_b, c_b = self.sequence_pass(sequence, h_0, c_0, reverse=True)
            h_b = h_b
            c_b = c_b
            # Concat both
            h = torch.cat((h_f, h_b), dim=2)
            c = torch.cat((c_f, c_b), dim=2)
        else:
            h, c = h_f, c_f
        n_directions = 2 if self.bidirectional else 1
        h_n = h[-1, :, :].view(n_directions, sequence.shape[1], self.hidden_size)
        c_n = c[-1, :, :].view(n_directions, sequence.shape[1], self.hidden_size)
        return h, (h_n, c_n)

    def sequence_pass(self, sequence, h_0, c_0, reverse=False):
        """
        DESCRIPTION:
        Method to go through the sequence in one way or the other. 
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
        batch_norms = [self.bn_hh, self.bn_ih, self.bn_c] if self.batch_norm else []
        if reverse:
            for i in range(sequence.shape[0]-1, -1, -1):
                time = -i + (sequence.shape[0]-1)
                h_t, c_t = self.cell(sequence[i].permute(1, 0), hs[-1], cs[-1],
                                    self.weight_ih_reverse, self.weight_hh_reverse,
                                    self.bias_reverse,
                                    # For the reverse case we use the same BN, just 
                                    # revese the sequence
                                    batch_norms, time)
                hs.append(h_t)
                cs.append(c_t)
            h = torch.flip(torch.stack(hs[1:], dim=0), dims=(0,)).permute(0, 2, 1)
            c = torch.flip(torch.stack(cs[1:], dim=0), dims=(0,)).permute(0, 2, 1)
        else:
            for i in range(sequence.shape[0]):
                time = i
                # Define the batch normalisations to use
                h_t, c_t = self.cell(sequence[i, :, :].permute(1, 0), hs[-1], cs[-1], 
                                    self.weight_ih, self.weight_hh, self.bias,
                                    batch_norms, time)
                hs.append(h_t)
                cs.append(c_t)
            h = torch.stack(hs[1:], dim=0).permute(0, 2, 1)
            c = torch.stack(cs[1:], dim=0).permute(0, 2, 1)
        return h, c

    def initialise_matrix(self, dims, method='orthogonal'):
        """
        DESCRIPTION:
        Function to initialise matrices based on the uniform or othogonal methods.
        :param dims: [list] dimensions of the matrix to initialise.
        :param params: [list] list with the parameters to initialise the matrices.
        :param method: [str] the method to initialise.
        :return: [torch.Tensor] the initialised matrix.
        """
        uniform_params = [(1 / self.hidden_size) ** 0.5] * 2
        orthogonal_params = [1]  # gain
        if len(dims) == 2:
            matrices = [torch.empty(int(dims[0] / 4), dims[1]) for _ in range(4)]
        else:
            matrices = [torch.empty(int(dims[0] / 4)) for _ in range(4)]
        if method == 'uniform':
            matrices = [init.uniform_(matrix, *uniform_params) for matrix in matrices]
        elif method == 'identity':
            [torch.eye(*matrix.shape, out=matrix) for matrix in matrices]
        elif method == 'zeros':
            [torch.zeros(*matrix.shape, out=matrix) for matrix in matrices]
        elif method == 'orthogonal':
            if len(dims) == 2:
                matrices = [init.orthogonal_(matrix, *orthogonal_params) for matrix in matrices]
            else:
                # Vector matrices (biases) cannot be orthogonal
                matrices = [init.uniform_(matrix, *uniform_params) for matrix in matrices]
        else:
            raise ValueError('Incorrect matrix initialisation method chosen.')
        return torch.cat(matrices, dim=0)


class LSTM(nn.Module):
    """
    DESCRIPTION:
    Final LSTM module built on the previous ones.
    """
    # Methods
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, reference=None, bidirectional=False, batch_norm=False, gamma=0.1):
        """
        Class constructor.
        :param input_size: [int] dimensionality of every item in the input sequence.
        :param hidden_size: [int] dimensionality of every sequence item in the hidden space.
        :param batch_first: [bool] if True, the input should be
        [batch_size, sequence_length, n_features], otherwise 
        [sequence_length, batch_size, n_features]
        :param num_layers: [int] number of layers that compound the model.
        :param reference: [torch.nn.LSTM] a model to import the weights from.
        :param bidirectional: [bool] flag to indicate if the layers building the model should
        be bidirectional or not.
        :param batch_norm: [bool] flag to indicate if there should be recurrent batch
        normalisation.
        :param gamma: [float] gamma value to initialse the batch norms.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.n_directions = 2 if bidirectional else 1
        # Create the layers
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(LSTMlayer(input_size, hidden_size, i, reference, bidirectional, batch_norm, gamma))
            else:
                self.layers.append(LSTMlayer(self.n_directions * hidden_size, hidden_size, i, reference, bidirectional, batch_norm, gamma))
    
    def forward(self, sequence):
        """
        DESCRIPTION:
        Forward pass.
        :param sequence: [torch.Tensor] sequence to feed the network. Dimensionality:
        [sequence_length, batch_size, input_size] or [batch_size, sequence_length, input_size]
        if bach_first == True.
        :return: the of torch.nn.LSTM. [torch.Tensor] the hidden state of the last layer,
        dimensionality: [sequence_length, batch_size, number_of_directions*hidden_size] or 
        [batch_size, sequence_length, number_of_directions*hidden_size] if batch_first == True.
        The final tuple are the last-neuron hidden and cell states, both of dimensionality:
        [number_of_directions*hidden_size, batch_size, hidden_size].
        """
        # Correct for the batch_first
        if self.batch_first:
            sequence = sequence.permute(1, 0, 2)
        h_n = []
        c_n = []
        output = sequence
        for layer in self.layers:
            output, (h_t, c_t) = layer(output)
            h_n.append(h_t)
            c_n.append(c_t)
        h_n = torch.stack(h_n, dim=0).view(self.n_directions * self.num_layers, -1, self.hidden_size)
        c_n = torch.stack(c_n, dim=0).view(self.n_directions * self.num_layers, -1, self.hidden_size)
        # Adapt the output to be batch first
        if self.batch_first:
            return output.permute(1, 0, 2), (h_n, c_n)    
        else:
            return output, (h_n, c_n) 


# Functions
def lstm_cell(x, h_t_1, c_t_1, weight_ih, weight_hh, bias, *args):
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
    ifgo = weight_hh @ h_t_1 + weight_ih @ x + bias
    i, f, g, o = torch.split(ifgo, int(weight_ih.shape[0] / 4), dim=0)
    c_t = torch.sigmoid(f) * c_t_1 + torch.sigmoid(i) * torch.tanh(g)
    h_t = torch.sigmoid(o) * torch.tanh(c_t)
    return h_t, c_t


def bnlstm_cell(x, h_t_1, c_t_1, weight_ih, weight_hh, bias, batch_norms, time):
    """
    DESCRIPTION:
    Function that represents the LSTM cell with batch normalisation. Adapted
    from: arXiv:1603.09025 [cs.LG]
    :param x: [torch.Tensor] cell input, value in the sequence for a time step.
    Dimensionality: [input_size, batch_size]. The second is 1 because it is
    always one item from the general sequence.
    :param h_t_1: [torch.Tensor] hidden state of the previous cell.
    Dimensionality: [hidden_size, batch_size].
    :param c_t_1: [torch.Tensor] cell state of the previous cell.
    Dimensionality: [hidden_size, batch_size].
    :param weight_ih: [torch.Tensor] layer weights to multiply by the input. 
    Dimensionality: [4 * hidden_size, input_size]
    :param weight_hh: [torch.Tensor] layer weights t multiply by the previous
    hidden state.
    Dimensionality: [4 * hidden_size, hidden_size]
    :param bias_ih: [torch.Tensor] layer bias of the input.
    Dimensionality: [4 * hidden_size, 1]
    :param bias_hh: [torch.Tensor] layer bias of the output.
    Dimensionality: [4 * hidden_size, 1]
    :param batch_norms: [list] vector to store the torch.BatchNorm1d that 
    we will apply, one per transformation and direction.
    :param time: [int] position in the sequence of the item computed.
    :return: hidden and cell states associated with this time step (h_t, c_t),
    both with dimensionality: [hidden_size, batch_size].
    """
    w_hh_by_h_t_1 = weight_hh.cuda() @ h_t_1.cuda()
    w_hh_by_h_t_1 = w_hh_by_h_t_1.permute(1, 0)
    w_ih_by_x = weight_ih.cuda() @ x.cuda()
    w_ih_by_x = w_ih_by_x.permute(1, 0)
    ifgo = batch_norms[0](w_hh_by_h_t_1, time).permute(1, 0) + \
            batch_norms[1](w_ih_by_x, time).permute(1, 0) + bias
    i, f, g, o = torch.split(ifgo, int(weight_ih.shape[0] / 4), dim=0)
    c_t = torch.sigmoid(f) * c_t_1.cuda() + torch.sigmoid(i) * torch.tanh(g)
    h_t = torch.sigmoid(o) * torch.tanh(batch_norms[2](c_t.permute(1, 0), time)).permute(1, 0)
    return h_t, c_t


    

