import numpy as np
import tensorflow as tf
import time
import datetime
from datetime import timedelta
import logging
from .config import *

def gen_data(n=BATCH_SIZE, block_len=BLOCK_LEN):
    return np.random.randint(0, 2, size=(n, block_len))

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

def binarize(bit):
    return tf.to_int32(tf.rint(bit))

def bsc(bit, p=ERR_PROB):
    return bit if np.random.random() >= ERR_PROB else tf.subtract(1, bit)

class TrainFormatter(logging.Formatter):
    def __init__(self):
        self.iter_time = time.time()
        self.epoch_time = time.time()
        self.epoch = 0
        super(TrainFormatter, self).__init__('%(asctime)s/%(name)s: %(levelname)s - %(message)s')

    def format(self, record):
        try:
            return super(TrainFormatter, self).format(record)
        except:
            return self.train_format(record, record.msg, record.args)

    def train_format(self, record, iteration, args):
        if len(args) > 1:
            self.epoch_time = time.time()
            self.epoch = args[1]
            epoch = True
        else:
            epoch = False
        err = args[0]

        iter_elapsed = timedelta(seconds=(time.time() - self.iter_time)).total_seconds()
        epoch_elapsed = timedelta(seconds=(time.time() - self.epoch_time)).total_seconds()
        self.iter_time = time.time()

        message = '[epoch:' + str(self.epoch) + '/step:' + str(iteration) + ']'
        message += 'Epoch Error:' if epoch else ' Error:'
        message += str(err)
        message += ' (Step: ' + str(iter_elapsed) + ' Epoch: ' + str(epoch_elapsed) + ')'

        record.msg = message
        record.args = ()

        return super(TrainFormatter, self).format(record)
