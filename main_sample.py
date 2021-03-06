import lasagne
import numpy as np
from models import def_net_arch
from bolognese import load_img_txt
from bolognese import compile_theano_expr
from bolognese import train, AnnealingPolicy, InitialStatus
from bolognese import PrintLog, AutoSnapshot, RememberBestWeights
from bolognese import DataAugmentation

NUM_EPOCHES = 5000
BATCH_SIZE = 2000
LEARNING_RATE = AnnealingPolicy('step', base_lr=0.01, gamma=0.96, step=500)
UPDATE_METHOD = lasagne.updates.nesterov_momentum
PRINT_LOG = PrintLog(log_file = './log_sample.txt')
AUTOSNAP = AutoSnapshot('./snapshot_sample', milestone = 200, lowerbound_trigger=0.99)
INIT = InitialStatus('./snapshot_sample/epoch_100.npz', start_iter_stage = 100)


print('Loading data...')
# data, label = load_img_txt('../mnist.txt', '../mnist/')
# indices = np.arange(len(data))
# np.random.shuffle(indices)
# dataset = dict(
#     X_train = [data[i] for i in indices[:50000]],
#     y_train = [label[i] for i in indices[:50000]],
#     X_valid = [data[i] for i in indices[50000:]],
#     y_valid = [label[i] for i in indices[50000:]]
#     )

with np.load('../test/mnist.npz', mmap_mode = None) as f:
    Data = f['Data']
    Label = f['Label']
dataset = dict(
    X_train = Data[:50000].reshape(50000, 1, 28, 28),
    y_train = Label[:50000],
    X_valid = Data[50000:].reshape(20000, 1, 28, 28),
    y_valid = Label[50000:]
    )

print('Compiling network...')
compiled_net = compile_theano_expr(def_net_arch, UPDATE_METHOD)

print('Starting training...')
train(
    compiled_net,
    dataset,
    NUM_EPOCHES,
    BATCH_SIZE,
    LEARNING_RATE,
    print_log = PRINT_LOG,
    autosnap = AUTOSNAP,
    init_stat = INIT,
    augmentation = None, # DataAugmentation(p=0.3, h_flip=True, v_flip=True, rotate=True)
    rememberbestweights = RememberBestWeights(key = 'valid_loss')
    )

