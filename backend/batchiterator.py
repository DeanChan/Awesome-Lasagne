from dataloader import parallel_load
import numpy as np

class BatchIterator(object):
    """
    Threaded batch iterator.
    for batch in BatchIterator(X, y, batchsize, shuffle=True, cache_size=5):
        X, y = batch
        ...

    X: numpy array or list of image paths
    y: numpy array or list of int numbers
    """

    def __init__(self, 
        inputs, 
        targets, 
        batchsize, 
        shuffle=True, 
        cache_size=5):
        super(BatchIterator, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
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
        if self.shuffle:
            indices = np.arange(len(self.inputs))
            np.random.shuffle(indices)

        if isinstance(self.inputs[0], np.ndarray):    
            for start_idx in range(0, max(len(self.inputs) - self.batchsize + 1, len(self.inputs)), self.batchsize):
                if self.shuffle:
                    excerpt = indices[start_idx:start_idx + self.batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + self.batchsize)
                yield self.inputs[excerpt], self.targets[excerpt]
        elif isinstance(self.inputs[0], str):
            for start_idx in range(0, max(len(self.inputs) - self.batchsize + 1, len(self.inputs)), self.batchsize):
                if self.shuffle:
                    excerpt = indices[start_idx:start_idx + self.batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + self.batchsize)
                yield parallel_load([self.inputs[i] for i in excerpt]), np.array([self.targets[i] for i in excerpt])




        