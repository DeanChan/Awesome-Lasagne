import sys
import os
import re
import lasagne
import numpy as np
from collections import OrderedDict
from tabulate import tabulate


class ansi:
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog:
    """
    This class is modified from 
    https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/handlers.py#-17-62
    """
    def __init__(self, log_file=None, print_interval=1):
        self.first_iteration = True
        self.log_file = log_file
        self.print_interval = print_interval

    def __call__(self, train_history):
        if train_history[-1]['epoch'] % self.print_interval == 0:
            to_print = self.table(train_history, self.first_iteration)
            
            if self.first_iteration: self.first_iteration = False 
            
            print(to_print)
            sys.stdout.flush()
            
            if self.log_file is not None:
                for s in to_print.split('\n'):
                    os.system('echo {} >> {}'.format(self.decolorize(s), self.log_file))
    
    @staticmethod
    def table(train_history, first_iteration):
        info = train_history[-1]

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.CYAN if info['train_loss_best'] else "",
                info['train_loss'],
                ansi.ENDC if info['train_loss_best'] else "",
                )),
            ('valid loss', "{}{:.5f}{}".format(
                ansi.GREEN if info['valid_loss_best'] else "",
                info['valid_loss'],
                ansi.ENDC if info['valid_loss_best'] else "",
                )),
            
            ])

        if 'valid_accuracy' in info:
            info_tabulate['valid acc'] = "{}{:.4f}{}".format(
                ansi.RED if info['valid_accuracy_best'] else "",
                info['valid_accuracy'],
                ansi.ENDC if info['valid_accuracy_best'] else "",
                )

        info_tabulate['dur'] = "{:.2f}s".format(info['dur'])

        tabulated = tabulate(
            [info_tabulate], headers="keys")

        out = ""
        if first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"

        out += tabulated.rsplit('\n', 1)[-1]
        return out

    @staticmethod
    def decolorize(string):
        color_pattern = r'\033\[\d+m'
        return re.sub(color_pattern, '', string)


class AutoSnapshot:
    """
    TO BE TESTED: @staricmethod or @classmethod of PrintLog.table and PrintLog.decolorize
    1. milestone: int, save model parameters every a specified interval
    2. lowerbound_trigger: float, save best parameters when accuracy is lager than it
    """
    def __init__(self, 
        path,
        milestone=200,
        lowerbound_trigger=0.99):
        self.path = path if path[-1] == '/' else path + '/'
        self.milestone = milestone
        self.lowerbound_trigger = lowerbound_trigger
        self.info_file = self.path + 'model_info.txt'
        self.first_iteration = True
        
    def __call__(self, model, train_history):
        info = train_history[-1]
        if (info['epoch'] % self.milestone == 0) or \
           (info['valid_accuracy_best'] and info['valid_accuracy'] >= self.lowerbound_trigger):
            self.dump_model(model, info['epoch'])
            self.snap_record(train_history)

    def dump_model(self, model, epoch):
        all_params = lasagne.layers.get_all_param_values(model)
        filename = self.path + 'epoch_{}.npz'.format(epoch)
        np.savez(filename, *all_params)

    def snap_record(self, train_history):
        model_info = PrintLog.table(train_history, self.first_iteration)
        if self.first_iteration: self.first_iteration = False 
        for s in model_info.split('\n'):
            os.system('echo {} >> {}'.format(PrintLog.decolorize(s), self.info_file))

