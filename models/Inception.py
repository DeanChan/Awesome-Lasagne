from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import LocalResponseNormalization2DLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify


def inception_3a(net):
    """
    bottom: "pool2/3x3_s2"
    top   : "inception_3a/output"
    """
    net['inception_3a/1x1'] = Conv2DLayer(net['pool2/3x3_s2'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_3a/3x3_reduce'] = Conv2DLayer(net['pool2/3x3_s2'],
        num_filters = 96,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_3a/3x3'] = Conv2DLayer(net['inception_3a/3x3_reduce'],
        num_filters = 128,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_3a/5x5_reduce'] = Conv2DLayer(net['pool2/3x3_s2'],
        num_filters = 16,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_3a/5x5'] = Conv2DLayer(net['inception_3a/5x5_reduce'],
        num_filters = 32,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_3a/pool'] = MaxPool2DLayer(net['pool2/3x3_s2'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_3a/pool_proj'] = Conv2DLayer(net['inception_3a/pool'],
        num_filters = 32,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_3a/output'] = ConcatLayer(
        [net['inception_3a/1x1'], 
        net['inception_3a/3x3'], 
        net['inception_3a/5x5'], 
        net['inception_3a/pool_proj']])

    return net


def inception_3b(net):
    """
    bottom: "inception_3a/output"
    top   : "inception_3b/output"
    """
    net['inception_3b/1x1'] = Conv2DLayer(net['inception_3a/output'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_3b/3x3_reduce'] = Conv2DLayer(net['inception_3a/output'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_3b/3x3'] = Conv2DLayer(net['inception_3b/3x3_reduce'],
        num_filters = 192,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_3b/5x5_reduce'] = Conv2DLayer(net['inception_3a/output'],
        num_filters = 32,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_3b/5x5'] = Conv2DLayer(net['inception_3b/5x5_reduce'],
        num_filters = 96,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_3b/pool'] = MaxPool2DLayer(net['inception_3a/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_3b/pool_proj'] = Conv2DLayer(net['inception_3b/pool'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_3b/output'] = ConcatLayer([
        net['inception_3b/1x1'], 
        net['inception_3b/3x3'], 
        net['inception_3b/5x5'], 
        net['inception_3b/pool_proj']])

    return net


def inception_4a(net):
    """
    bottom: "pool3/3x3_s2"
    top   : "inception_4a/output"
    """
    net['inception_4a/1x1'] = Conv2DLayer(net['pool3/3x3_s2'],
        num_filters = 192,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4a/3x3_reduce'] = Conv2DLayer(net['pool3/3x3_s2'],
        num_filters = 96,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4a/3x3'] = Conv2DLayer(net['inception_4a/3x3_reduce'],
        num_filters = 208,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_4a/5x5_reduce'] = Conv2DLayer(net['pool3/3x3_s2'],
        num_filters = 16,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4a/5x5'] = Conv2DLayer(net['inception_4a/5x5_reduce'],
        num_filters = 48,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_4a/pool'] = MaxPool2DLayer(net['pool3/3x3_s2'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_4a/pool_proj'] = Conv2DLayer(net['inception_4a/pool'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4a/output'] = ConcatLayer([
        net['inception_4a/1x1'], 
        net['inception_4a/3x3'], 
        net['inception_4a/5x5'], 
        net['inception_4a/pool_proj']])

    return net


def inception_4b(net):
    """
    bottom: "inception_4a/output"
    top   : "inception_4b/output"
    """
    net['inception_4b/1x1'] = Conv2DLayer(net['inception_4a/output'],
        num_filters = 160,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4b/3x3_reduce'] = Conv2DLayer(net['inception_4a/output'],
        num_filters = 112,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4b/3x3'] = Conv2DLayer(net['inception_4b/3x3_reduce'],
        num_filters = 224,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_4b/5x5_reduce'] = Conv2DLayer(net['inception_4a/output'],
        num_filters = 24,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4b/5x5'] = Conv2DLayer(net['inception_4b/5x5_reduce'],
        num_filters = 64,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_4b/pool'] = MaxPool2DLayer(net['inception_4a/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_4b/pool_proj'] = Conv2DLayer(net['inception_4b/pool'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4b/output'] = ConcatLayer([
        net['inception_4b/1x1'], 
        net['inception_4b/3x3'], 
        net['inception_4b/5x5'], 
        net['inception_4b/pool_proj']])

    return net


def inception_4c(net):
    """
    bottom: "inception_4b/output"
    top   : "inception_4c/output"
    """
    net['inception_4c/1x1'] = Conv2DLayer(net['inception_4b/output'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4c/3x3_reduce'] = Conv2DLayer(net['inception_4b/output'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4c/3x3'] = Conv2DLayer(net['inception_4c/3x3_reduce'],
        num_filters = 256,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_4c/5x5_reduce'] = Conv2DLayer(net['inception_4b/output'],
        num_filters = 24,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4c/5x5'] = Conv2DLayer(net['inception_4c/5x5_reduce'],
        num_filters = 64,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_4c/pool'] = MaxPool2DLayer(net['inception_4b/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_4c/pool_proj'] = Conv2DLayer(net['inception_4c/pool'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4c/output'] = ConcatLayer([
        net['inception_4c/1x1'], 
        net['inception_4c/3x3'], 
        net['inception_4c/5x5'], 
        net['inception_4c/pool_proj']])

    return net


def inception_4d(net):
    """
    bottom: "inception_4c/output"
    top   : "inception_4d/output"
    """
    net['inception_4d/1x1'] = Conv2DLayer(net['inception_4c/output'],
        num_filters = 112,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4d/3x3_reduce'] = Conv2DLayer(net['inception_4c/output'],
        num_filters = 144,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4d/3x3'] = Conv2DLayer(net['inception_4d/3x3_reduce'],
        num_filters = 288,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_4d/5x5_reduce'] = Conv2DLayer(net['inception_4c/output'],
        num_filters = 32,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4d/5x5'] = Conv2DLayer(net['inception_4d/5x5_reduce'],
        num_filters = 64,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_4d/pool'] = MaxPool2DLayer(net['inception_4c/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_4d/pool_proj'] = Conv2DLayer(net['inception_4d/pool'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4d/output'] = ConcatLayer([
        net['inception_4d/1x1'], 
        net['inception_4d/3x3'], 
        net['inception_4d/5x5'], 
        net['inception_4d/pool_proj']])

    return net


def inception_4e(net):
    """
    bottom: "inception_4d/output"
    top   : "inception_4e/output"
    """
    net['inception_4e/1x1'] = Conv2DLayer(net['inception_4d/output'],
        num_filters = 256,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4e/3x3_reduce'] = Conv2DLayer(net['inception_4d/output'],
        num_filters = 160,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4e/3x3'] = Conv2DLayer(net['inception_4e/3x3_reduce'],
        num_filters = 320,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_4e/5x5_reduce'] = Conv2DLayer(net['inception_4d/output'],
        num_filters = 32,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_4e/5x5'] = Conv2DLayer(net['inception_4e/5x5_reduce'],
        num_filters = 128,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_4e/pool'] = MaxPool2DLayer(net['inception_4d/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_4e/pool_proj'] = Conv2DLayer(net['inception_4e/pool'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_4e/output'] = ConcatLayer([
        net['inception_4e/1x1'], 
        net['inception_4e/3x3'], 
        net['inception_4e/5x5'], 
        net['inception_4e/pool_proj']])

    return net


def inception_5a(net):
    """
    bottom: "pool4/3x3_s2"
    top   : "inception_5a/output"
    """
    net['inception_5a/1x1'] = Conv2DLayer(net['pool4/3x3_s2'],
        num_filters = 256,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_5a/3x3_reduce'] = Conv2DLayer(net['pool4/3x3_s2'],
        num_filters = 160,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_5a/3x3'] = Conv2DLayer(net['inception_5a/3x3_reduce'],
        num_filters = 320,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_5a/5x5_reduce'] = Conv2DLayer(net['pool4/3x3_s2'],
        num_filters = 32,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_5a/5x5'] = Conv2DLayer(net['inception_5a/5x5_reduce'],
        num_filters = 128,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_5a/pool'] = MaxPool2DLayer(net['pool4/3x3_s2'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_5a/pool_proj'] = Conv2DLayer(net['inception_5a/pool'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_5a/output'] = ConcatLayer([
        net['inception_5a/1x1'], 
        net['inception_5a/3x3'], 
        net['inception_5a/5x5'], 
        net['inception_5a/pool_proj']])

    return net


def inception_5b(net):
    """
    bottom: "inception_5a/output"
    top   : "inception_5b/output"
    """
    net['inception_5b/1x1'] = Conv2DLayer(net['inception_5a/output'],
        num_filters = 384,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_5b/3x3_reduce'] = Conv2DLayer(net['inception_5a/output'],
        num_filters = 192,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_5b/3x3'] = Conv2DLayer(net['inception_5b/3x3_reduce'],
        num_filters = 384,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )

    net['inception_5b/5x5_reduce'] = Conv2DLayer(net['inception_5a/output'],
        num_filters = 48,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['inception_5b/5x5'] = Conv2DLayer(net['inception_5b/5x5_reduce'],
        num_filters = 128,
        filter_size = 5,
        flip_filters = False,
        pad = 2,
        nonlinearity = rectify
        )

    net['inception_5b/pool'] = MaxPool2DLayer(net['inception_5a/output'],
        pool_size = 3,
        stride = 1,
        pad = 1,
        )
    net['inception_5b/pool_proj'] = Conv2DLayer(net['inception_5b/pool'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )

    net['inception_5b/output'] = ConcatLayer([
        net['inception_5b/1x1'], 
        net['inception_5b/3x3'], 
        net['inception_5b/5x5'], 
        net['inception_5b/pool_proj']])

    return net


def build_network():
    net = {}
#-----------------------------------------------------------------------------------#   
    net['data'] = InputLayer((None, 1, 128, 128))
    net['conv1/7x7_s2'] = Conv2DLayer(net['data'],
        num_filters = 64, 
        filter_size = 7, 
        flip_filters = False,
        stride = 2, 
        pad = 3,
        nonlinearity = rectify
        )
    net['pool1/3x3_s2'] = MaxPool2DLayer(net['conv1/7x7_s2'],
        pool_size = 3,
        stride = 2,
        ignore_border = False
        )
    net['pool1/norm1'] = LocalResponseNormalization2DLayer(net['pool1/3x3_s2'],
        n = 5,
        alpha = 0.0001,
        beta = 0.75
        )
    net['conv2/3x3_reduce'] = Conv2DLayer(net['pool1/norm1'],
        num_filters = 64,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['conv2/3x3'] = Conv2DLayer(net['conv2/3x3_reduce'],
        num_filters = 192,
        filter_size = 3,
        flip_filters = False,
        pad = 1,
        nonlinearity = rectify
        )
    net['conv2/norm2'] = LocalResponseNormalization2DLayer(net['conv2/3x3'],
        n = 5,
        alpha = 0.0001,
        beta = 0.75
        )
    net['pool2/3x3_s2'] = MaxPool2DLayer(net['conv2/norm2'],
        pool_size = 3,
        stride = 2,
        ignore_border = False
        )
#-----------------------------------------------------------------------------------#   
    net = inception_3a(net)
    net = inception_3b(net)
    net['pool3/3x3_s2'] = MaxPool2DLayer(net['inception_3b/output'],
        pool_size = 3,
        stride = 2,
        ignore_border = False
        )
    net = inception_4a(net)
    net = inception_4b(net)
    net = inception_4c(net)
    net = inception_4d(net)
    net = inception_4e(net)
    net['pool4/3x3_s2'] = MaxPool2DLayer(net['inception_4e/output'],
        pool_size = 3,
        stride = 2,
        ignore_border = False
        )
    net = inception_5a(net)
    net = inception_5b(net)
#-----------------------------------------------------------------------------------#     
    net['loss1/ave_pool'] = Pool2DLayer(net['inception_4a/output'],
        pool_size = 5,
        stride = 3,
        mode = 'average_exc_pad',
        # ignore_border = False
        )
    net['loss1/conv'] = Conv2DLayer(net['loss1/ave_pool'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['loss1/fc'] = DenseLayer(net['loss1/conv'],
        num_units = 1024,
        nonlinearity = rectify
        )
    net['loss1/fc'] = DropoutLayer(net['loss1/fc'], p = 0.7)
    net['loss1/classifier'] = DenseLayer(net['loss1/fc'],
        num_units = 10575,
        nonlinearity = None
        ) ##                                                                 FEATURE_1
    net['loss1/softmax'] = NonlinearityLayer(net['loss1/classifier'],
        nonlinearity = softmax
        ) ##                                                         LOSS_WEIGHT = 0.3
#-----------------------------------------------------------------------------------#     
    net['loss2/ave_pool'] = Pool2DLayer(net['inception_4d/output'],
        pool_size = 5,
        stride = 3,
        mode = 'average_exc_pad',
        # ignore_border = False
        )
    net['loss2/conv'] = Conv2DLayer(net['loss2/ave_pool'],
        num_filters = 128,
        filter_size = 1,
        flip_filters = False,
        nonlinearity = rectify
        )
    net['loss2/fc'] = DenseLayer(net['loss2/conv'],
        num_units = 1024,
        nonlinearity = rectify
        )
    net['loss2/fc'] = DropoutLayer(net['loss2/fc'], p = 0.7)
    net['loss2/classifier'] = DenseLayer(net['loss2/fc'],
        num_units = 10575,
        nonlinearity = None
        ) ##                                                                 FEATURE_2
    net['loss2/softmax'] = NonlinearityLayer(net['loss2/classifier'],
        nonlinearity = softmax
        ) ##                                                         LOSS_WEIGHT = 0.3
    ##                                                               ACCURACY TOP1/TOP5
#-----------------------------------------------------------------------------------#     
    net['pool5/7x7_s1'] = Pool2DLayer(net['inception_5b/output'],
        pool_size = 3,
        stride = 1,
        mode = 'average_exc_pad',
        # ignore_border = False
        )
    net['pool5/7x7_s1'] = DropoutLayer(net['pool5/7x7_s1'], p = 0.4)
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
        num_units = 10575,
        nonlinearity = None
        ) ##                                                                 FEATURE_3
    net['loss3/softmax'] = NonlinearityLayer(net['loss3/classifier'],
        nonlinearity = softmax
        ) ##                                                         LOSS_WEIGHT = 1.0

    return net