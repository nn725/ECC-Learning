import tensorflow as tf

from argparse import ArgumentParser
from src.model import SimpleAgents, AdversaryAgents, IndependentAgents
from src.config import *

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('type', type=str, dest='sys_type',
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

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        if options.sys_type == 'simple':
            agents_class = SimpleAgent
        elif options.sys_type == 'adversary':
            agents_class = AdversaryAgents
        else:
            agents_class = IndependentAgents

        agents = agents_class(sess, block_len=options.block_len,
                msg_len=options.msg_len, batch_size=options.batch_size,
                epochs=options.epochs, learning_rate=options.rate)

        agents.train()

if __name__ == '__main__':
    main()
