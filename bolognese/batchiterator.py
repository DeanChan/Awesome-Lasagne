import numpy as np
from scipy.misc import imrotate
from multiprocessing import Pool
from .dataloader import parallel_load


class BatchIterator(object):
    """
    Threaded batch iterator.
    for batch in BatchIterator(X, y, batchsize, shuffle=True, augment=None, cache_size=5):
        X, y = batch
        ...

    X: numpy array or list of image paths
    y: numpy array or list of int numbers
    augment: None or callable
    """

    def __init__(self, 
        inputs, 
        targets, 
        batchsize, 
        shuffle=True,
        augment=None, 
        cache_size=5):
        super(BatchIterator, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.augment = augment
        self.batchsize = batchsize
        self.cache_size = cache_size

    def __iter__(self):
        return self.threaded_generator(self.iterate_minibatches())

    def threaded_generator(self, generator):
        '''
        This code is written by jan Schluter
        copied from https://github.com/benanne/Lasagne/issues/12
        '''
        import Queue
        queue = Queue.Queue(maxsize = self.cache_size)
        sentinel = object()

        def producer():
            for item in generator:
                queue.put(item)
            queue.put(sentinel)

        import threading
        thread = threading.Thread(target = producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not sentinel:
            yield item
            queue.task_done()
            item = queue.get()

    def iterate_minibatches(self):
        assert len(self.inputs) == len(self.targets)
        indices = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(indices)

        if isinstance(self.inputs[0], np.ndarray):
            for start_idx in range(0, max(len(self.inputs) - self.batchsize + 1, len(self.inputs)), self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize] 
                out_data = self.augmentation(self.inputs[excerpt])                          
                yield out_data, self.targets[excerpt]
        
        elif isinstance(self.inputs[0], str):
            for start_idx in range(0, max(len(self.inputs) - self.batchsize + 1, len(self.inputs)), self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]
                out_data = self.augmentation(parallel_load([self.inputs[i] for i in excerpt]))
                yield out_data, np.array([self.targets[i] for i in excerpt])

    def augmentation(self, X):
        return self.augment(X) if self.augment is not None else X


class DataAugmentation(object):
    """
    Chose a subset of input images randomly and replace them with disordered ones.
    f_noise = DataAugmentation(p=0.3, h_flip=True, v_flip=True, rotate=False)
    X_noise = f_noise(X)

    X.shape = (batchsize, channels, img_height, img_width)
    X_noise.shape = X.shape
    """
    def __init__(self, 
        p = 0.3,
        h_flip = True,
        v_flip = True,
        rotate = False,):
        super(DataAugmentation, self).__init__()
        self.p = p
        self.h_flip = h_flip 
        self.v_flip = v_flip
        self.rotate = rotate

    def __call__(self, X):
        if self.h_flip:
            indices = np.random.choice(X.shape[0], X.shape[0]*self.p, replace=False)
            X[indices] = self.horizonal_flip(X[indices])
        if self.v_flip:
            indices = np.random.choice(X.shape[0], X.shape[0]*self.p, replace=False)
            X[indices] = self.vertical_flip(X[indices])
        if self.rotate:
            indices = np.random.choice(X.shape[0], X.shape[0]*self.p, replace=False)
            X[indices] = self.rotate_img(X[indices])
        return X

    def horizonal_flip(self, X):
        return X[:, :, :, ::-1]

    def vertical_flip(self, X):
        return X[:, :, ::-1, :]

    def rotate_img(self, X):       
        
        def theta():
            return np.random.random() * 90 - 45 # random rotation degree from [-45, 45]        
        
        for i in range(X.shape[0]):
            deg = theta()
            for j in range(X.shape[1]):
                X[i, j, :, :] = imrotate(X[i, j, :, :], deg)

        return X


        