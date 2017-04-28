import numpy as np
import tensorflow as tf
# from .config import *
from config import *

def gen_data(n=BATCH_SIZE, block_len=BLOCK_LEN):
    return np.random.randint(0, 2, size=(n, block_len))

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

def binarize(bit):
    return tf.to_int32(tf.rint(bit))

def bsc(bit, p=ERR_PROB):
    return bit if np.random.random() >= ERR_PROB else tf.subtract(1, bit)
