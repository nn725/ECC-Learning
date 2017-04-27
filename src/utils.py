import numpy as np
import tensorflow as tf
from config import *

def gen_data(n=BATCH_SIZE, block_length=BLOCK_LEN):
    return np.random.randint(0, 2, size=(n, block_length)*2-1)

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

def bsc(bits, p=ERR_PROB):
    out = bits.copy()
    indices = (np.random(len(bits)) <= p)
    out[indices] = 1 ^ out[indices]
    return out
