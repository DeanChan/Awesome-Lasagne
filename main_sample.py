import lasagne
import numpy as np
from models import def_net_arch
from bolognese import load_img_txt
from bolognese import compile_theano_expr
from bolognese import train
from bolognese import PrintLog, AutoSnapshot

NUM_EPOCHES = 5000
BATCH_SIZE = 200
BASE_LEARNING_RATE = 0.01
UPDATE_METHOD = lasagne.updates.nesterov_momentum
PRINT_LOG = PrintLog(log_file = './log_sample.txt')
AUTOSNAP = AutoSnapshot('./snapshot_sample', milestone = 200, lowerbound_trigger=0.99)


print('Loading data...')
data, label = load_img_txt('../mnist.txt', '../mnist/')
indices = np.arange(len(data))
np.random.shuffle(indices)
dataset = dict(
    X_train = [data[i] for i in indices[:50000]],
    y_train = [label[i] for i in indices[:50000]],
    X_valid = [data[i] for i in indices[50000:]],
    y_valid = [label[i] for i in indices[50000:]]
    )

print('Compiling network...')
compiled_net = compile_theano_expr(def_net_arch, UPDATE_METHOD)

print('Starting training...')
train(
    compiled_net,
    dataset,
    NUM_EPOCHES,
    BASE_LEARNING_RATE,
    BATCH_SIZE,
    PRINT_LOG,
    AUTOSNAP,
    augmentation = None
    )

