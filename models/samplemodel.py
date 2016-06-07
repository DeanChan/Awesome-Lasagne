from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import DenseLayer, DropoutLayer


def def_net_arch(input_var = None):
    net = {}
    net['input'] = InputLayer(shape = (None, 1, 28, 28),
        input_var = input_var)

    net['conv1'] = Conv2DLayer(net['input'], 
        num_filters = 32,
        filter_size = (5,5),
        nonlinearity = rectify,
        )

    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size = (2,2))

    net['conv2'] = Conv2DLayer(net['pool1'],
        num_filters = 32,
        filter_size = (5,5),
        nonlinearity = rectify,
        )

    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size = (2,2))

    net['fc1'] = DenseLayer(net['pool2'],
        num_units = 256,
        nonlinearity = rectify
        )

    net['fc1_dropout'] = DropoutLayer(net['fc1'], p = 0.5)

    net['softmax_out'] = DenseLayer(net['fc1_dropout'],
        num_units = 10,
        nonlinearity = softmax
        )

    return net