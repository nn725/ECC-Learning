import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from argparse import ArgumentParser
from src.model import SimpleAgents, HammingAgents, AdversaryAgents, IndependentAgents
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
            choices=('simple', 'hamming', 'adversary', 'independent'),
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

    parser.add_argument('--num_change', type=int, dest='num_change',
            help='max number of flipped bits', metavar='NUM_CHANGE',
            default=NUM_CHANGE)

    parser.add_argument('--batch-size', type=int, dest='batch_size',
            help='batch size', metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('-d', dest='debug', help='debug',
            action='store_true')

    return parser

def plotErrors():
    sns.set_style('darkgrid')
    #num_ch[i] is max errors for each num change from 0 through 4 for num_change=i
    max_rec_errors_num_ch = []
    max_bin_errors_num_ch = []
    for i in range(0,4):
        NUM_CHANGE = i
        with tf.Session() as sess:
            agents = SimpleAgents(sess, block_len=BLOCK_LEN, msg_len=MSG_LEN,
                    inter_len=INTER_LEN, batch_size=BATCH_SIZE, epochs=100,
                    learning_rate=0.01, num_change=i, level=None)

            all_rec_errors, all_bin_errors = agents.train()
            max_rec_errors = []
            max_bin_errors = []
            for i in range(len(all_rec_errors)):
                max_rec_errors.append(max(all_rec_errors[i]))
                max_bin_errors.append(max(all_bin_errors[i]))

            max_rec_errors_num_ch.append(max_rec_errors)
            max_bin_errors_num_ch.append(max_bin_errors)

    num_ch_to_max_rec_err = []
    num_ch_to_max_bin_err = []
    for j in range(0,4):
        #x axis
        num_ch_rec_err_i = []
        num_ch_bin_err_i = []
        for i in range(0,4):
            #which index to extract
            num_ch_rec_err_i.append(max_rec_errors_num_ch[i][j])
            num_ch_bin_err_i.append(max_bin_errors_num_ch[i][j])

        num_ch_to_max_rec_err.append(num_ch_rec_err_i)
        num_ch_to_max_bin_err.append(num_ch_bin_err_i)

    plt.title('Reconstruction errors across multiple bit flips')
    for i in range(len(num_ch_to_max_rec_err)):
        plt.plot(num_ch_to_max_rec_err[i])
    plt.legend(['0 bit flips', '1 bit flip', '2 bit flips', '3 bit flips', '4 bit flips'])
    plt.xlabel('Max number of bits flipped for training')
    plt.ylabel('Average decoding error')
    xint = range(0, 4)
    plt.xticks(xint)
    plt.show()
    plt.savefig('rec_errors_across_multiple_bit_flips.eps', format='eps', dpi=1000)

    plt.title('Binarized errors across multiple bit flips')
    for i in range(len(num_ch_to_max_bin_err)):
        plt.plot(num_ch_to_max_bin_err[i])
    plt.legend(['0 bit flips', '1 bit flip', '2 bit flips', '3 bit flips', '4 bit flips'])
    plt.xlabel('Max number of bits flipped for training')
    plt.ylabel('Average decoding error')
    xint = range(0, 4)
    plt.xticks(xint)
    plt.show()
    plt.savefig('bin_errors_across_multiple_bit_flips.eps', format='eps', dpi=1000)



def main():
    logger.debug('Building parser')
    parser = build_parser()
    logger.debug('Parsing args')
    options = parser.parse_args()

    #plotErrors()

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
        if options.sys_type == 'hamming':
            logger.info('Using HammingAgents')
            agents_class = HammingAgents
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
