import tensorflow as tf

from . import config
from . import utils
# from layers import conv_layer

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns


import logging

mod_logger = logging.getLogger(__name__)

#initial plan, set up Alice and Bob nets and the commpy BSC channel
class BaseAgents(object):
    def __init__(self, sess, block_len=config.BLOCK_LEN, msg_len=config.MSG_LEN,
                inter_len=config.INTER_LEN, batch_size=config.BATCH_SIZE,
                epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE, 
                num_change=config.NUM_CHANGE, level=None):

        self.sess = sess

        if not level:
            level = logging.INFO

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(level)
        ch = logging.StreamHandler()
        fh = logging.FileHandler('train_log')
        ch.setFormatter(utils.TrainFormatter())
        fh.setFormatter(utils.TrainFormatter())
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        self.logger.info('MSG_LEN = ' + str(msg_len))
        self.msg_len = msg_len
        self.logger.info('BLOCK_LEN = ' + str(block_len))
        self.block_len = block_len
        self.logger.info('INTER_LEN = ' + str(inter_len))
        self.inter_len = inter_len
        self.N = block_len
        self.logger.info('BATCH_SIZE = ' + str(batch_size))
        self.batch_size = batch_size
        self.logger.info('EPOCHS = ' + str(epochs))
        self.epochs = epochs
        self.logger.info('LEARNING_RATE = ' + str(learning_rate))
        self.learning_rate = learning_rate
        self.logger.info('NUM_CHANGE = ' + str(num_change))
        self.num_change = num_change

        self.logger.info('BUILDING MODEL')
        self.build_model()

    def build_model(self):
        pass

    def train(self):
        pass

    def save_model(self, filename):
        self.trans_saver.save(self.sess, filename+'_transmitter')
        self.rec_saver.save(self.sess, filename+'_receiver')


class SimpleAgents(BaseAgents):
    def __init__(self, *args, **kwargs):
        super(SimpleAgents, self).__init__(*args, **kwargs)

    def build_model(self):
        self.l1_transmitter = utils.init_weights("transmitter_w_l1", [self.N, self.inter_len])
        self.l2_transmitter = utils.init_weights("transmitter_w_l2", [self.inter_len, self.msg_len])
        self.l1_receiver = utils.init_weights("receiver_w_l1", [self.msg_len, self.inter_len])
        self.l2_receiver = utils.init_weights("receiver_w_l2", [self.inter_len, self.N])

        self.msg = tf.placeholder("float", [None, self.N])

        self.trans_saver = tf.train.Saver([self.l1_transmitter, self.l2_transmitter])
        self.rec_saver = tf.train.Saver([self.l1_receiver, self.l2_receiver])

        # self.trans_saver = tf.train.Saver([self.l1_transmitter])
        # self.rec_saver = tf.train.Saver([self.l1_receiver])

        #transmitter network
        #FC layer (block_len (N) x N) -> FC Layer (N x msg_len) -> Output Layer (msg_len x msg_len)
        #(not used yet) FC layer -> Conv layer
        self.transmitter_hidden_1 = tf.tanh(tf.matmul(self.msg, self.l1_transmitter))
        # self.transmitter_hidden_1 = tf.matmul(self.msg, self.l1_transmitter)
        self.transmitter_output = tf.squeeze(tf.tanh(tf.matmul(self.transmitter_hidden_1, self.l2_transmitter)))
        #self.transmitter_output = tf.squeeze(tf.tanh(tf.matmul(self.transmitter_hidden_2, self.l3_transmitter)))
        # self.transmitter_output = tf.verify_tensor_all_finite(self.transmitter_output,
        #         'transmitter output not finite')

        # #alternate
        # # FC -> 2 conv layers
        # self.transmitter_hidden_1 = tf.nn.sigmoid(tf.matmul(self.msg, self.l1_transmitter))
        # self.transmitter_output = tf.squeeze(conv_layer(self.transmitter_hidden_1, "transmitter"))

        # self.channel_input = utils.binarize(self.transmitter_output)
        # self.channel_output = utils.bsc(self.channel_input)
        self.channel_output = self.transmitter_output

        #reciever network
        #FC layer (msg_len x msg_len) -> FC Layer (msg_len x N) -> Output layer (N x N)
        #(not used yet) Conv Layer -> FC Layer
        self.receiver_hidden_1 = tf.tanh(tf.matmul(self.channel_output, self.l1_receiver))
        self.receiver_output = tf.squeeze(tf.tanh(tf.matmul(self.receiver_hidden_1, self.l2_receiver)))
        #self.receiver_output = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.receiver_hidden_2, self.l3_receiver)))
        #self.receiver_output = tf.squeeze(tf.tanh(tf.matmul(self.receiver_hidden_2, self.l3_receiver)))
        self.receiver_output_binary = utils.binarize(self.receiver_output)

        # #alternate
        # # 2 conv -> FC
        # self.receiver_conv = conv_layer(self.channel_output, "receiver")
        # self.receiver_output = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.receiver_conv, self.l1_receiver)))

    def train(self):
        #Loss functions
        self.rec_loss = tf.reduce_mean(tf.abs(self.msg - self.receiver_output)/2)
        self.bin_loss = tf.reduce_mean(tf.abs(self.msg - self.receiver_output_binary)/2)
        # self.bin_loss = tf.Print(self.bin_loss, [self.msg, self.channel_output, self.receiver_output_binary], summarize=4)
        #get training variables
        self.train_vars = tf.trainable_variables()
        self.trans_or_rec_vars = [var for var in self.train_vars if 'transmitter_' in var.name or 'receiver_' in var.name]

        #optimizers
        self.rec_optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(
                self.rec_loss, var_list=self.trans_or_rec_vars)

        self.rec_errors = []
        self.bin_errors = []

        #training
        tf.global_variables_initializer().run()
        # tf.initialize_all_variables().run()
        for i in range(self.epochs):
            iterations = 500
            self.logger.info('Training Epoch: ' + str(i))
            rec_loss, bin_loss = self._train(iterations, i)
            self.logger.info(iterations, rec_loss, bin_loss, i)
            self.rec_errors.append(rec_loss)
            self.bin_errors.append(bin_loss)

        self.plot_errors()

    def _train(self, iterations, epoch):
        rec_error = 1.0
        bin_error = 1.0

        bs = self.batch_size

        for i in range(iterations):
            msg = utils.gen_data(n=bs, block_len=self.block_len)

            _, decode_err, bin_loss = self.sess.run([self.rec_optimizer,
                self.rec_loss, self.bin_loss], feed_dict={self.msg: msg})
            self.logger.debug(i, decode_err, bin_loss)
            rec_error = min(rec_error, decode_err)
            bin_error = min(bin_error, bin_loss)

        return rec_error, bin_error

    def plot_errors(self):
        sns.set_style('darkgrid')
        plt.plot(self.rec_errors)
        plt.plot(self.bin_errors)
        plt.legend(['loss', 'binary error'])
        plt.xlabel('Epoch')
        plt.ylabel('Lowest decoding error achieved')
        plt.show()


class AdversaryAgents(BaseAgents):
    def __init__(self, *args, **kwargs):
        super(AdversaryAgents, self).__init__(*args, **kwargs)

    def build_model(self):
        pass

    def train(self):
        pass

class IndependentAgents(BaseAgents):
    def __init__(self, *args, **kwargs):
        super(IndependentAgents, self).__init__(*args, **kwargs)

    def build_model(self):
        pass

    def train(self):
        pass
