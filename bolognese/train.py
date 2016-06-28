import lasagne
import time
import numpy as np
from .batchiterator import BatchIterator
from .batchiterator import DataAugmentation
from .monitor import RememberBestWeights, PrintLog, AutoSnapshot

class LrPolicy:
    """
    method = 'fixed', 'step', 'exp', 'inv', 'poly', 'sigmoid'
    """
    def __init__(self,
        method,
        max_iter = 30000,
        gamma = 0.9,
        step = 1000,
        power = 1.1
        ):
        self.max_iter = float(max_iter)
        self.gamma = gamma
        self.step = float(step)
        self.power = power 
        table = {'fixed'   : self.fixed,
                 'step'    : self.step_,
                 'exp'     : self.exp,
                 'inv'     : self.inv,
                 'poly'    : self.poly,
                 'sigmoid' : self.sigmoid}

        self.decay = table[method]

    def __call__(self, base_lr, epoch):
        return self.decay(base_lr, epoch)

    def fixed(self, base_lr, epoch):
        return base_lr

    def step_(self, base_lr, epoch):
        return base_lr * self.gamma**(np.floor(epoch / self.step))

    def exp(self, base_lr, epoch):
        return base_lr * self.gamma**(epoch)

    def inv(self, base_lr, epoch):
        return base_lr * (1 + self.gamma * epoch)**(-self.power)

    def poly(self, base_lr, epoch):
        return base_lr * (1 - epoch / self.max_iter)**(self.power)

    def sigmoid(self, base_lr, epoch):
        return base_lr * (1.0 / (1 + np.exp(-self.gamma * (epoch - self.step))))


def train(
    compiled_net,
    dataset,
    num_epochs,
    batch_size,
    learning_rate,
    lr_policy = LrPolicy('fixed'),
    print_log = PrintLog(log_file = './log_sample.txt'),
    autosnap = AutoSnapshot('./snapshot_sample'),
    extra_update_arg_list = [],
    start_iter_stage = 0,
    batch_iterator =BatchIterator,
    shuffle_batch = True,
    augmentation = DataAugmentation(p=0.3, h_flip=True, v_flip=True, rotate=False),
    rememberbestweights = RememberBestWeights(key = 'valid_loss'),
    ):

    try:
        best_train_loss = np.inf
        best_valid_loss = np.inf
        best_valid_acc = 0
        for epoch in range(start_iter_stage, num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            curr_lr = lr_policy(learning_rate, epoch)
            for batch in batch_iterator(dataset['X_train'], dataset['y_train'], batch_size, shuffle_batch, augmentation):
                inputs, targets = batch
                train_err += compiled_net['train'](inputs, targets, curr_lr, *extra_update_arg_list)
                train_batches += 1
            train_loss = train_err / train_batches
            best_train_loss = train_loss if train_loss < best_train_loss else best_train_loss

            val_err = 0 
            val_acc = 0 
            val_batches = 0
            for batch in batch_iterator(dataset['X_valid'], dataset['y_valid'], batch_size, False, None):
                inputs, targets = batch
                err, acc = compiled_net['test'](inputs, targets)
                val_err += err 
                val_acc += acc
                val_batches += 1
            valid_loss = val_err / val_batches
            valid_acc = val_acc / val_batches
            best_valid_loss = valid_loss if valid_loss < best_valid_loss else best_valid_loss
            best_valid_acc = valid_acc if valid_acc > best_valid_acc else best_valid_acc

            info = dict(
                epoch = epoch,
                train_loss = train_loss,
                valid_loss = valid_loss,
                train_loss_best = train_loss == best_train_loss,
                valid_loss_best = valid_loss == best_valid_loss,
                valid_accuracy = valid_acc,
                valid_accuracy_best = valid_acc == best_valid_acc,
                current_lr = curr_lr, 
                dur = time.time() - start_time
                )

            print_log(info)
            autosnap(compiled_net['net_arch']['softmax_out'], info)
            rememberbestweights(compiled_net['net_arch']['softmax_out'], info)

        rememberbestweights.store(
            rememberbestweights.best_weights,
            autosnap.path,
            rememberbestweights.best_weights_epoch,
            rememberbestweights.best_weights_loss
            )

    except KeyboardInterrupt:
        rememberbestweights.store(
            rememberbestweights.best_weights,
            autosnap.path,
            rememberbestweights.best_weights_epoch,
            rememberbestweights.best_weights_loss
            )
        pass


