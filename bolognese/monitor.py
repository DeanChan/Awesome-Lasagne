import sys
import os
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
    def __init__(self, log_save_path=None):
        self.first_iteration = True
        self.log_save_path = log_save_path

    def __call__(self, train_history):
        to_print = self.table(train_history)
        print(to_print)
        sys.stdout.flush()
        if self.log_save_path is not None:
            for s in to_print.split('\n'):
                os.system('echo {} >> {}'.format(s, self.log_save_path))

    def table(self, train_history):
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
            info_tabulate['valid acc'] = "{}{:.2f}{}".format(
                ansi.RED if info['valid_accuracy_best'] else "",
                info['valid_accuracy'],
                ansi.ENDC if info['valid_accuracy_best'] else "",
                )

        info_tabulate['dur'] = "{:.2f}s".format(info['dur'])

        tabulated = tabulate(
            [info_tabulate], headers="keys", floatfmt='.5f')

        out = ""
        if self.first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"
            self.first_iteration = False

        out += tabulated.rsplit('\n', 1)[-1]
        return out