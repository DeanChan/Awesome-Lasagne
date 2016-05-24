from multiprocessing import Pool
from scipy.misc import imread
import numpy as np
import theano

def load_img_txt(txt_path, img_path):
    """
    X, y = load_img_txt(txt_path, img_path)
    
    txt_path: ./relative/path/to/img label
    img_path: /absolute/path/to/imgs/
    return: 
            X: list of absolute paths to every image
            y: list of int numbers
    """
    data, labels = [], []
    img_path = img_path if img_path[-1] == '/' else img_path + '/'
    with open(txt_path, 'r') as f:
        raw_strings = f.readlines()
    for raw_line in raw_strings:
        rl_path, label = raw_line.split(' ')
        data.append(img_path + rl_path[2:])
        labels.append(int(label))

    return data, labels


def parallel_load(path_list, num_processes=4):
    """
    Load a bunch of images in parallel.
    """
    num_processes = min(num_processes, len(path_list))
    pool = Pool(num_processes)
    imgs = pool.map(imread, path_list)
    pool.close()
    pool.join()
    
    out_data = np.array(imgs, dtype=theano.config.floatX)
    if out_data.ndim == 3:
        return out_data[:, np.newaxis, :, :]
    else:
        return out_data.transpose((0, 3, 1, 2))




#DEPRECATED
# def load_np_array(path):
#     """
#     Load numpy array in a lazy style,
#     with np.mmap_mode = 'r'
#     file: xxx.npz/xxx.npy include numpy array only
#     """
#     data_dict = np.load(path, mmap_mode = 'r')
#     return data_dict['Data'], data_dict['Label']