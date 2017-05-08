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

        self.biases = {
                'transmitter_b1': tf.Variable(tf.random_normal([self.inter_len])),
                'transmitter_b2': tf.Variable(tf.random_normal([self.msg_len])),
                'receiver_b1': tf.Variable(tf.random_normal([self.inter_len])),
                'receiver_b2': tf.Variable(tf.random_normal([self.N]))
                }

        self.msg = tf.placeholder("float", [None, self.N])

        self.trans_saver = tf.train.Saver([self.l1_transmitter, self.l2_transmitter])
        self.rec_saver = tf.train.Saver([self.l1_receiver, self.l2_receiver])


        self.transmitter_hidden_1 = tf.tanh(tf.add(tf.matmul(self.msg, self.l1_transmitter), self.biases['transmitter_b1']))
        self.transmitter_output = tf.squeeze(tf.tanh(tf.add(tf.matmul(self.transmitter_hidden_1, self.l2_transmitter), self.biases['transmitter_b2'])))


        self.channel_input = utils.binarize(self.transmitter_output)
        self.channel_output = utils.bsc(self.channel_input)


        self.receiver_hidden_1 = tf.tanh(tf.add(tf.matmul(self.channel_output, self.l1_receiver), self.biases['receiver_b1']))
        self.receiver_output = tf.squeeze(tf.tanh(tf.add(tf.matmul(self.receiver_hidden_1, self.l2_receiver), self.biases['receiver_b2'])))

        self.receiver_output_binary = utils.binarize(self.receiver_output)

    def train(self):
        #Loss functions
        self.rec_loss = tf.reduce_mean(tf.abs(self.msg - self.receiver_output)/2)
        self.bin_loss = tf.reduce_mean(tf.abs(self.msg - self.receiver_output_binary)/2)
        # self.bin_loss = tf.Print(self.bin_loss, [self.msg], first_n=16, summarize=4)
        #get training variables
        self.train_vars = tf.trainable_variables()
        self.trans_or_rec_vars = [var for var in self.train_vars if 'transmitter_' in var.name or 'receiver_' in var.name]

        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(self.learning_rate, global_step, 500*self.batch_size*self.epochs, 1)
        #optimizers
        self.rec_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
                self.rec_loss+0.5*self.bin_loss, var_list=self.trans_or_rec_vars, global_step=global_step)

        self.rec_errors = []
        self.bin_errors = []

        #training
        tf.global_variables_initializer().run()
        for i in range(self.epochs):
            iterations = 500
            self.logger.info('Training Epoch: ' + str(i))
            rec_loss, bin_loss = self._train(iterations, i)
            self.logger.info(iterations, rec_loss, bin_loss, i)
            self.rec_errors.append(rec_loss)
            self.bin_errors.append(bin_loss)

        self.plot_errors()
        self.comparePerformance(self.num_change-1)

    def _train(self, iterations, epoch):
        rec_error = 0.0
        bin_error = 0.0

        bs = self.batch_size

        for i in range(iterations):
            msg = utils.gen_data(n=bs, block_len=self.block_len)

            _, decode_err, bin_loss = self.sess.run([self.rec_optimizer,
                self.rec_loss, self.bin_loss], feed_dict={self.msg: msg})
            self.logger.debug(i, decode_err, bin_loss)
            rec_error = max(rec_error, decode_err)
            bin_error = max(bin_error, bin_loss)

        return rec_error, bin_error

    def plot_errors(self):
        sns.set_style('darkgrid')
        plt.plot(self.rec_errors)
        plt.plot(self.bin_errors)
        plt.legend(['loss', 'binary error'])
        plt.xlabel('Epoch')
        plt.ylabel('Lowest decoding error achieved')
        plt.show()

    def comparePerformance(self, test_num_change):
        #build same model with weights and test on
        msg = tf.placeholder("float", [None, self.N])


        transmitter_hidden_1 = tf.tanh(tf.add(tf.matmul(msg, self.l1_transmitter), self.biases['transmitter_b1']))
        transmitter_output = tf.squeeze(tf.tanh(tf.add(tf.matmul(transmitter_hidden_1, self.l2_transmitter), self.biases['transmitter_b2'])))


        channel_input = utils.binarize(transmitter_output)
        channel_output = utils.bsc_forward(channel_input, self.msg_len, self.batch_size, test_num_change)


        receiver_hidden_1 = tf.tanh(tf.add(tf.matmul(channel_output, self.l1_receiver), self.biases['receiver_b1']))
        receiver_output = tf.squeeze(tf.tanh(tf.add(tf.matmul(receiver_hidden_1, self.l2_receiver), self.biases['receiver_b2'])))

        receiver_output_binary = utils.binarize(receiver_output)

        rec_loss = tf.reduce_mean(tf.abs(msg - receiver_output)/2)
        bin_loss = tf.reduce_mean(tf.abs(msg - receiver_output_binary)/2)

        bs = self.batch_size

        rec_error = 0.0
        bin_error = 0.0

        rec_errors = []
        bin_errors = []

        trained_rec_errors = []
        trained_bin_errors = []

        iterations = 20

        for i in range(iterations):
            msgs = utils.gen_data(n=bs, block_len=self.block_len)
            decode_err, binary_loss = self.sess.run([rec_loss, bin_loss], feed_dict={msg: msgs})
            trained_decode_err, trained_binary_loss = self.sess.run([self.rec_loss, self.bin_loss], feed_dict={self.msg: msgs})

            rec_errors.append(decode_err)
            bin_errors.append(binary_loss)
            trained_rec_errors.append(trained_decode_err)
            trained_bin_errors.append(trained_binary_loss)

        self.plotComparison(trained_rec_errors, trained_bin_errors, rec_errors, bin_errors)


    def plotComparison(self, trained_rec_errors, trained_bin_errors, comp_rec_errors, comp_bin_errors):
        sns.set_style('darkgrid')
        plt.plot(trained_rec_errors)
        plt.plot(trained_bin_errors)
        plt.plot(comp_rec_errors)
        plt.plot(comp_bin_errors)
        plt.legend(['tr_rec_errors', 'tr_bin_errors', 'comp_rec_errors', 'comp_bin_errors'])
        plt.xlabel('iteration')
        plt.ylabel('lowest decoding error achieved')
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
