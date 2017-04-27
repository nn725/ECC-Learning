import tensorflow as tf
from .config import *
from .utils import *

#initial plan, set up Alice and Bob nets and the commpy BSC channel
class BaseAgents(object):
    def __init__(self, sess, block_len=BLOCK_LEN, msg_len=MSG_LEN, 
                batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, 
                learning_rate=LEARNING_RATE):

        self.sess = sess
        self.msg_len = msg_len
        self.block_len = block_len
        self.N = block_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self):
        pass

    def train(self):
        pass


class SimpleAgents(BaseAgents):
    def __init__(self, *args, **kwargs):
        super(SimpleAgents, self).__init__(*args, **kwargs)

    def build_model(self):
        self.l1_transmitter = init_weights("transmitter_w_l1", [self.N, self.N])
        self.l2_transmitter = init_weights("transmitter_w_l2", [self.N, self.msg_len])
        self.l3_transmitter = init_weights("transmitter_w_l3", [self.msg_len, self.msg_len])
        self.l1_receiver = init_weights("receiver_w_l1", [self.msg_len, self.msg_len])
        self.l2_receiver = init_weights("receiver_w_l2", [self.msg_len, self.N])
        self.l3_receiver = init_weights("receiver_w_l3", [self.N, self.N])

        self.msg = tf.placeholder("float", [None, self.N])

        #transmitter network
        #FC layer (block_len (N) x N) -> FC Layer (N x msg_len) -> Output Layer (msg_len x msg_len)
        ##(not used yet) FC layer -> Conv layer
        self.transmitter_hidden_1 = tf.nn.sigmoid(tf.matmul(self.msg, self.l1_transmitter))
        self.transmitter_hidden_2 = tf.nn.sigmoid(tf.matmul(self.transmitter_hidden_1, self.l2_transmitter))
        self.transmitter_output = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.transmitter_hidden_2, self.l3_transmitter)))

        self.channel_input = tf.map_fn(binarize, self.transmitter_output, dtype=tf.int8)
        self.channel_output = tf.map_fn(bsc, self.channel_input)

        #reciever network
        #FC layer (msg_len x msg_len) -> FC Layer (msg_len x N) -> Output layer (N x N)
        ##(not used yet) Conv Layer -> FC Layer
        self.receiver_hidden_1 = tf.nn.sigmoid(tf.matmul(self.channel_output, self.l1_receiver))
        self.receiver_hidden_2 = tf.nn.sigmoid(tf.matmul(self.receiver_hidden_1, self.l2_receiver))
        self.receiver_output = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.receiver_hidden_2, self.l3_receiver)))

    def train(self):
        #Loss functions
        self.rec_loss = tf.reduce_mean(tf.abs(self.msg - self.receiver_output))

        #get training variables
        self.train_vars = tf.training_variables()
        self.trans_or_rec_vars = [var for var in self.train_vars if 'transmitter_' in var.name or 'receiver_' in var.name]

        #optimizers
        self.rec_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
                                self.rec_loss, var_list=self.trans_or_rec_vars)

        self.rec_errors = []

        #training
        tf.initialize_all_variables().run()
        for i in range(self.epochs):
            iterations = 2000

            print('Training Transmitter and Receiver, Epoch:', i + 1)
            rec_loss = self._train(iterations)
            self.rec_errors.append(rec_loss)


    def _train(self, iterations):
        rec_error = 1.0

        bs = self.batch_size

        for i in range(iterations):
            msg = gen_data(n=bs, block_len=self.block_len)

            _, decode_err = self.sess.run([self.rec_optimizer, self.rec_loss],
                                               feed_dict={self.msg: msg})

            rec_error = min(rec_error, decode_err)

        return rec_error


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
