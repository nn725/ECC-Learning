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
        self.l1_receiver = init_weights("receiver_w_l1", [self.msg_len, self.N])
        self.l2_receiver = init_weights("receiver_w_l2", [self.N, self.N])

        self.msg = self.placeholder("float", [None, self.N])

        #transmitter network
        #FC layer (block_len (N) x N) -> FC Layer (N x msg_len)
        ##(not used yet) FC layer -> Conv layer
        self.transmitter_hidden_1 = tf.nn.sigmoid(tf.matmul(self.msg, self.l1_transmitter))
        self.transmitter_hidden_2 = tf.nn.sigmoid(tf.matmul(self.transmitter_hidden_1, self.l2_transmitter))
        self.transmitter_output = tf.squeeze(self.transmitter_hidden_2)

        #reciever network
        #FC layer (msg_len x N) -> FC Layer (N x N)
        ##(not used yet) Conv Layer -> FC Layer
        self.receiver_hidden_1 = tf.nn.sigmoid(tf.matmul(self.transmitter_output, self.l1_receiver))
        self.receiver_hidden_2 = tf.nn.sigmoid(tf.matmul(self.receiver_hidden_1, self.l2_receiver))
        self.receiver_output = tf.squeeze(self.receiver_hidden_2)


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
