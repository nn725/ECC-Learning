import numpy as np
import tensorflow as tf
from .config import *

def gen_data(n=BATCH_SIZE, block_length=BLOCK_LEN):
    return np.random.randint(0, 2, size=(n, block_length)*2-1)

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

def binarize(bit):
    return 0 if bit <= 0.5 else 1

def bsc(bit, p=ERR_PROB):
    assert type(bit) == int
    assert bit <= 1
    assert bit >= 0

    return bit if np.random.random() >= ERR_PROB else 1 ^ bit
