import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
from tltorch.factorized_tensors import CPTensor, TTTensor, TuckerTensor, FactorizedTensor
# from tltorch.factorized_layers import _ensure_array, _ensure_list, factorization_shape_to_kernel_shape, kernel_shape_to_factorization_shape, kernel_to_tensor, tensor_to_kernel
import tltorch.factorized_layers
import warnings


def _ensure_list(order, value):
    """Ensures that `value` is a list of length `order`

    If `value` is an int, turns it into a list ``[value]*order``
    """
    if isinstance(value, int):
        return [value]*order
    assert len(value) == order
    return value

def _ensure_array(layers_shape, order, value, one_per_order=True):
    """Ensures that `value` is an array

    Parameters
    ----------
    layers_shape : tuple
        shape of the layer (n_weights)
    order : int
        order of the convolutional layer
    value : np.ndarray or int
        value to be checked
    one_per_order : bool, optional
        if true, then we must have one value per mode of the convolution
        otherwise, a single value per factorized layer is needed
        by default True

    Returns
    -------
    np.ndarray
        if one_per_order, of shape layers_shape
        otherwise, of shape (*layers_shape, order)
    """
    if one_per_order:
        target_shape = layers_shape + (order, )
    else:
        target_shape = layers_shape

    if isinstance(value, np.ndarray):
        assert value.shape == target_shape
        return value

    if isinstance(value, int):
        array = np.ones(target_shape, dtype=int)*value
    else:
        assert len(value) == order
        array = np.ones(target_shape, dtype=int)
        array[..., :] = value
    return array

def kernel_shape_to_factorization_shape(factorization, kernel_shape):
    """Returns the shape of the factorized weights to create depending on the factorization    
    """
    # For the TT case, the decomposition has a different shape than the kernel.
    if factorization.lower() == 'tt':
        kernel_shape = list(kernel_shape)
        out_channel = kernel_shape.pop(0)
        kernel_shape.append(out_channel)
        return tuple(kernel_shape)

    # Other decompositions require no modification
    return kernel_shape

def factorization_shape_to_kernel_shape(factorization, factorization_shape):
    """Returns a convolutional kernel shape rom a factorized tensor shape
    """
    if factorization.lower() == 'tt':
        kernel_shape = list(factorization_shape)
        out_channel = kernel_shape.pop(-1)
        kernel_shape = [out_channel] + kernel_shape
        return tuple(kernel_shape)
    return factorization_shape

def kernel_to_tensor(factorization, kernel):
    """Returns a convolutional kernel ready to be factorized
    """
    if factorization.lower() == 'tt':
        kernel = tl.moveaxis(kernel, 0, -1)
    return kernel

def tensor_to_kernel(factorization, tensor):
    """Returns a kernel from a tensor factorization
    """
    if factorization.lower() == 'tt':
        tensor = tl.moveaxis(tensor, -1, 0)
    return tensor

def _get_factorized_conv(factorization, implementation='factorized'):
    # if implementation == 'reconstructed':
        # return convolve
    if isinstance(factorization, CPTensor):
        if implementation == 'factorized':
            return cp_conv_transpose
        # elif implementation == 'mobilenet':
            # return cp_conv_mobilenet
    # elif isinstance(factorization, TuckerTensor):
    #     return tucker_conv
    # elif isinstance(factorization, TTTensor):
    #     return tt_conv
    # raise ValueError(f'Got unknown type {factorization}')

class FactorizedConvTranspose(nn.Module):
    """Create a factorized convolution of arbitrary order
    """
    _version = 1
    
    def __init__(self, in_channels, out_channels, kernel_size, order=None,
                 stride=1, padding=0, dilation=1, bias=False, has_bias=False, n_layers=1, output_padding=0,
                 factorization='cp', rank='same', implementation='factorized', fixed_rank_modes=None):
        super().__init__()

        # Check that order and kernel size are well defined and match
        if isinstance(kernel_size, int):
            if order is None:
                raise ValueError(f'If int given for kernel_size, order (dimension of the convolution) should also be provided.')
            if not isinstance(order, int) or order <= 0:
                raise ValueError(f'order should be the (positive integer) order of the convolution'
                                 f'but got order={order} of type {type(order)}.')
            else:
                kernel_size = (kernel_size, ) * order
        else:
            kernel_size = tuple(kernel_size)
            order = len(kernel_size)
        
        self.order = order
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.implementation = implementation
        self.input_rank = rank
        self.n_layers = n_layers
        self.factorization = factorization
        self.output_padding = output_padding

        # Shape to insert if multiple layers are parametrized
        if isinstance(n_layers, int):
            if n_layers == 1:
                layers_shape = ()
            else:
                layers_shape = (n_layers, )
        else:
            layers_shape = n_layers
        self.layers_shape = layers_shape
    
        # tensor of values for each parametrized conv
        self.padding = _ensure_array(layers_shape, order, padding)
        self.stride = _ensure_array(layers_shape, order, stride)
        self.dilation = _ensure_array(layers_shape, order, dilation)
        self.has_bias = _ensure_array(layers_shape, order, has_bias, one_per_order=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(*layers_shape, out_channels))
        else:
            self.register_parameter('bias', None)

        if isinstance(factorization, FactorizedTensor):
            self.weight = factorization
            kernel_shape = factorization_shape_to_kernel_shape(factorization._name, factorization.shape)
        else:
            kernel_shape = (out_channels, in_channels) + kernel_size
            # Some factorizations require permuting the dimensions, handled by kernel_shape_to_factorization_shape
            kernel_shape = kernel_shape_to_factorization_shape(factorization, kernel_shape)
            # In case we are parametrizing multiple layers
            factorization_shape = layers_shape + kernel_shape

            # For Tucker decomposition, we may want to not decomposed spatial dimensions
            if fixed_rank_modes is not None:
                if factorization.lower() != 'tucker':
                    warnings.warn(f'Got fixed_rank_modes={fixed_rank_modes} which is only used for factorization=tucker but got factorization={factorization}.')
                elif fixed_rank_modes== 'spatial':
                    fixed_rank_modes = list(range(2 + len(layers_shape), 2+len(layers_shape)+order))

            self.weight = FactorizedTensor.new(factorization_shape, rank=rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes)

        self.rank = self.weight.rank
        self.shape = self.weight.shape
        self.kernel_shape = kernel_shape
        # We pre-select the forward function to not waste time doing the check at each forward pass
        self.forward_fun = _get_factorized_conv(self.weight, self.implementation)
        self.forward_fun = cp_conv_transpose
 
    def forward(self, x, indices=0):
        # Single layer parametrized
        if self.n_layers == 1:
            if indices == 0:
                return self.forward_fun(x, self.weight(), bias=self.bias, stride=self.stride, 
                                        padding=self.padding, dilation=self.dilation, output_padding=self.output_padding)
            else:
                raise ValueError(f'Only one convolution was parametrized (n_layers=1) but tried to access {indices}.')

        # Multiple layers parameterized
        if isinstance(self.n_layers, int):
            if not isinstance(indices, int):
                raise ValueError(f'Expected indices to be in int but got indices={indices}'
                                 f', but this conv was created with n_layers={self.n_layers}.')
        elif len(indices) != len(self.n_layers):
            raise ValueError(f'Got indices={indices}, but this conv was created with n_layers={self.n_layers}.')
        
        bias = self.bias[indices] if self.has_bias[indices] else None
        return self.forward_fun(x, self.weight(indices), bias=bias, stride=self.stride[indices], 
                                padding=self.padding[indices], dilation=self.dilation[indices])

    def set(self, indices, stride=1, padding=0, dilation=1, bias=None):
        """Sets the parameters of the conv self[indices]
        """
        self.padding[indices] = _ensure_list(self.order, padding)
        self.stride[indices] = _ensure_list(self.order, stride)
        self.dilation[indices] = _ensure_list(self.order, dilation)
        if bias is not None:
            self.bias.data[indices] = bias.data
            self.has_bias[indices] = True

    
    def __getitem__(self, indices):
        return self.get_conv(indices)

    @classmethod
    def from_factorization(cls, factorization, implementation='factorized',
                           stride=1, padding=0, dilation=1, bias=None, n_layers=1):
        kernel_shape = factorization_shape_to_kernel_shape(factorization._name, factorization.shape)
        
        if n_layers == 1:
            out_channels, in_channels, *kernel_size = kernel_shape
        elif isinstance(n_layers, int):
            layer_size, out_channels, in_channels, *kernel_size = kernel_shape
            assert layer_size == n_layers
        else:
            layer_size = kernel_shape[:len(n_layers)]
            out_channels, in_channels, *kernel_size = kernel_shape[len(n_layers):]

        order = len(kernel_size)

        instance = cls(in_channels, out_channels, kernel_size, order=order, implementation=implementation, 
                       padding=padding, stride=stride, bias=(bias is not None), n_layers=n_layers,
                       factorization=factorization, rank=factorization.rank)

        if bias is not None:
            instance.bias.data = bias

        return instance

    @classmethod 
    def from_conv(cls, conv_layer, rank='same', implementation='reconstructed', factorization='CP', 
                  decompose_weights=True, decomposition_kwargs=dict(), fixed_rank_modes=None, **kwargs):
        """Create a Factorized convolution from a regular convolutional layer
        
        Parameters
        ----------
        conv_layer : torch.nn.ConvND
        rank : rank of the decomposition, default is 'same'
        implementation : str, default is 'reconstructed'
        decomposed_weights : bool, default is True
            if True, the convolutional kernel is decomposed to initialize the factorized convolution
            otherwise, the factorized convolution's parameters are initialized randomly
        decomposition_kwargs : dict 
            parameters passed directly on to the decompoosition function if `decomposed_weights` is True
        
        Returns
        -------
        New instance of the factorized convolution with equivalent weightss

        Todo
        ----
        Check that the decomposition of the given convolution and cls is the same.
        """
        padding = conv_layer.padding
        out_channels, in_channels, *kernel_size = conv_layer.weight.shape
        stride = conv_layer.stride[0]
        bias = conv_layer.bias is not None

        instance = cls(in_channels, out_channels, kernel_size, 
                       factorization=factorization, implementation=implementation, rank=rank, 
                       padding=padding, stride=stride, fixed_rank_modes=fixed_rank_modes, bias=bias, **kwargs)

        if decompose_weights:
            if conv_layer.bias is not None:
                instance.bias.data = conv_layer.bias.data
        
            with torch.no_grad():
                kernel_tensor = kernel_to_tensor(factorization, conv_layer.weight.data)
                instance.weight.init_from_tensor(kernel_tensor, **decomposition_kwargs)

        return instance

    @classmethod
    def from_conv_list(cls, conv_list, rank='same', implementation='reconstructed', factorization='cp',
                       decompose_weights=True, decomposition_kwargs=dict(), **kwargs):
        conv_layer = conv_list[0]
        padding = conv_layer.padding
        out_channels, in_channels, *kernel_size = conv_layer.weight.shape
        stride = conv_layer.stride[0]
        bias = True

        instance = cls(in_channels, out_channels, kernel_size, implementation=implementation, rank=rank, factorization=factorization,
                       padding=padding, stride=stride, bias=bias, n_layers=len(conv_list), fixed_rank_modes=None, **kwargs)

        if decompose_weights:
            with torch.no_grad():
                weight_tensor = torch.stack([kernel_to_tensor(factorization, layer.weight.data) for layer in conv_list])
                instance.weight.init_from_tensor(weight_tensor, **decomposition_kwargs)

        for i, layer in enumerate(conv_list):
            instance.set(i, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=layer.bias)
            # instance.padding[i] = _ensure_list(instance.order, layer.padding)
            # instance.stride[i] = _ensure_list(instance.order, layer.stride)
            # instance.dilation[i] = _ensure_list(instance.order, layer.dilation)

        return instance

    def transduct(self, kernel_size, mode=0, padding=0, stride=1, dilation=1, fine_tune_transduction_only=True):
        """Transduction of the factorized convolution to add a new dimension

        Parameters
        ----------
        kernel_size : int
            size of the additional dimension
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)
        padding : int, default is 0
        stride : int: default is 1

        Returns
        -------
        self
        """
        if fine_tune_transduction_only:
            for param in self.parameters():
                param.requires_grad = False

        mode += len(self.layers_shape)
        self.order += 1
        padding = np.ones(self.layers_shape + (1, ), dtype=int)*padding
        stride = np.ones(self.layers_shape + (1, ), dtype=int)*stride
        dilation = np.ones(self.layers_shape + (1, ), dtype=int)*dilation

        self.padding = np.concatenate([self.padding[..., :mode], padding, self.padding[..., mode:]], len(self.layers_shape))
        self.stride = np.concatenate([self.stride[..., :mode], stride, self.stride[..., mode:]], len(self.layers_shape))
        self.dilation = np.concatenate([self.dilation[..., :mode], dilation, self.dilation[..., mode:]], len(self.layers_shape))

        self.kernel_size = self.kernel_size[:mode] + (kernel_size,) + self.kernel_size[mode:]
        self.kernel_shape = self.kernel_shape[:mode+2] + (kernel_size,) + self.kernel_shape[mode+2:]

        # Just to the frame-wise conv if adding time
        if isinstance(self.weight, CPTensor):
            new_factor = torch.zeros(kernel_size, self.weight.rank)
            new_factor[kernel_size//2, :] = 1
            transduction_mode = mode + 2
        elif isinstance(self.weight, TTTensor):
            new_factor = None
            transduction_mode = mode + 1
        else:
            transduction_mode = mode + 2
            new_factor = None

        self.weight = self.weight.transduct(kernel_size, transduction_mode, new_factor)

        return self

    def extra_repr(self):
        s = (f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'
             f', rank={self.rank}, order={self.order}')
        if self.n_layers == 1:
            s += ', '
            if self.stride.tolist() != [1] * self.order:
                s += f'stride={self.stride.tolist()}, '
            if self.padding.tolist() != [0] * self.order:
                s += f'padding={self.padding.tolist()}, '
            if self.dilation.tolist() != [1] * self.order:
                s += f'dilation={self.dilation.tolist()}, '
            if self.bias is None:
                s += f'bias=False'
            return s

        for idx in np.ndindex(self.n_layers):
            s += f'\n * Conv{idx}: '            
            if self.stride[idx].tolist() != [1] * self.order:
                s += f'stride={self.stride[idx].tolist()}, '
            if self.padding[idx].tolist() != [0] * self.order:
                s += f'padding={self.padding[idx].tolist()}, '
            if self.dilation[idx].tolist() != [1] * self.order:
                s += f'dilation={self.dilation[idx].tolist()}, '
            if self.bias is None:
                s += f'bias=False'
        return s


def cp_conv_transpose(x, cp_tensor, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Perform a factorized CP transposed convolution

    Parameters
    ----------
    x : torch.tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)
    cp_tensor : CP tensor
        CP tensor representing the decomposed convolutional kernel
    bias : torch.tensor, optional
        Bias tensor to be added to the output
    stride : int or tuple, optional
        Stride of the transposed convolution
    padding : int or tuple, optional
        Padding added to both sides of the input
    output_padding : int or tuple, optional
        Additional padding added to one side of the output
    dilation : int or tuple, optional
        Spacing between kernel elements

    Returns
    -------
    torch.tensor
        Output tensor after applying transposed CP convolution
    """
    shape = cp_tensor.shape
    rank = cp_tensor.rank
    batch_size = x.shape[0]
    order = len(shape) - 2

    if isinstance(padding, int):
        padding = (padding, )*order
    if isinstance(stride, int):
        stride = (stride, )*order
    if isinstance(dilation, int):
        dilation = (dilation, )*order
    if isinstance(output_padding, int):
        output_padding = (output_padding, )*order



    print(f'x input shape: {x.shape}')
    x = x.reshape((batch_size, x.shape[1], -1)).contiguous()
    # print(x.shape)
    print(1)
    print(tl.transpose(cp_tensor.factors[1]).unsqueeze(2).transpose(0, 1).shape)
    x = F.conv_transpose1d(x, tl.transpose(cp_tensor.factors[1]).unsqueeze(2).transpose(0, 1))

    x_shape = list(x.shape)
    x_shape[1] = rank
    x = x.reshape(x_shape)
    print(2)

    print(x.shape)
    print(f'stride: {stride[0]}')
    print(f'padding: {padding[0]}')
    print(f'kernel_size: {cp_tensor.factors[2].shape}')
    print(f'output_padding: {output_padding[0]}')
    
    # Transposed convolutions over non-channels dimensions
    for i in range(order):
        kernel = tl.transpose(cp_tensor.factors[i+2]).unsqueeze(1)
        print(f'kernel shape: {kernel.shape}')
        x = F.conv_transpose1d(
            x.contiguous(), kernel, stride=stride[i], padding=padding[i], 
            output_padding=output_padding[i], groups=rank
        )
    print(3)

    print(x.shape)

    # Convert back number of channels from rank to out_channels using tensor contraction
    x = x.reshape((batch_size, x.shape[1], -1))
    # print(x.shape)
    final_kernel = cp_tensor.factors[0].unsqueeze(2).transpose(0, 1)  # Make sure dimensions are [out_channels, rank, kernel_size]
    x = F.conv_transpose1d(
        x * cp_tensor.weights.unsqueeze(1).unsqueeze(0), 
        final_kernel, bias=bias, stride=1
    )
    print(4)
    print(x.shape)



    x_shape[1] = x.shape[1]  # Update to out_channels
    x_shape[2] = x.shape[2]
    x = x.reshape(x_shape)
    print(5)
    print(x.shape)

    return x