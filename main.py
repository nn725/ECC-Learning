import tensorflow as tf

from argparse import ArgumentParser
from src.model import SimpleAgents, AdversaryAgents, IndependentAgents
from src.config import *

from sys import version_info
from tensorflow.python import debug as tf_debug

input_fn = input if version_info[0] > 2 else raw_input

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('sys_type', type=str,
            choices=('simple', 'adversary', 'independent'),
            help='which type of system to use',
            metavar='TYPE', default='simple')

    parser.add_argument('--block-len', type=int, dest='block_len',
            help='block length', metavar='BLOCK_LEN', default=BLOCK_LEN)

    parser.add_argument('--msg-len', type=int, dest='msg_len',
            help='message length', metavar='MSG_LEN', default=MSG_LEN)

    parser.add_argument('--epochs', type=int, dest='epochs',
            help='number of epochs', metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--rate', type=float, dest='rate',
            help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--batch-size', type=int, dest='batch_size',
            help='batch size', metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('-d', dest='debug', help='debug',
            action='store_true')

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        if options.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if options.sys_type == 'simple':
            agents_class = SimpleAgents
        elif options.sys_type == 'adversary':
            agents_class = AdversaryAgents
        else:
            agents_class = IndependentAgents

        agents = agents_class(sess, block_len=options.block_len,
                msg_len=options.msg_len, batch_size=options.batch_size,
                epochs=options.epochs, learning_rate=options.rate)

        agents.train()
        save = input_fn("Save weights? (Y/n) ")
        if not save:
            return
        filename = input_fn("Filename: ")
        agents.save_model(filename)

if __name__ == '__main__':
    main()
