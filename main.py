import tensorflow as tf

from argparse import ArgumentParser
from src.model import SimpleAgents, AdversaryAgents, IndependentAgents
from src.config import *

from sys import version_info
from tensorflow.python import debug as tf_debug

input_fn = input if version_info[0] > 2 else raw_input

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler
fh = logging.FileHandler('log')
fh.setLevel(logging.DEBUG)
# create stream handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s/%(name)s: %(levelname)s - %(message)s')
# set formatters
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add handlers
logger.addHandler(ch)
logger.addHandler(fh)

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

    parser.add_argument('--inter-len', type=int, dest='inter_len',
            help='internal length', metavar='INTER_LEN', default=INTER_LEN)

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
    logger.debug('Building parser')
    parser = build_parser()
    logger.debug('Parsing args')
    options = parser.parse_args()

    with tf.Session() as sess:
        level = logging.INFO
        if options.debug:
            logger.info('Starting debug')
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            level = logging.DEBUG
        if options.sys_type == 'simple':
            logger.info('Using SimpleAgents')
            agents_class = SimpleAgents
        elif options.sys_type == 'adversary':
            logger.info('Using AdversaryAgents')
            agents_class = AdversaryAgents
        else:
            logger.info('Using IndependentAgents')
            agents_class = IndependentAgents

        logger.info('Building model')
        agents = agents_class(sess, block_len=options.block_len,
                msg_len=options.msg_len, inter_len=options.inter_len,
                batch_size=options.batch_size, epochs=options.epochs,
                learning_rate=options.rate, level=level)

        logger.info('Training')
        agents.train()
        logger.info('Done training')
        save = input_fn("Save weights? (y/n) ")
        while save not in ['y', 'n']:
            save = input_fn('Please enter y or n ')
        if save == 'n':
            logger.debug('Did not save weights')
            logger.info('Done')
            return
        filename = input_fn("Filename: ")
        logger.debug('Saving weights to ' + filename)
        agents.save_model(filename)
        logger.info('Done')

if __name__ == '__main__':
    main()
