# Moga Patricia - Ciobanu Sergiu-Tudor

import argparse
import os
import logzero
import logging

from dialtask.utils import load_conf, run_for_n_iterations, DialTaskFormatter
from dialtask.conversation_handler import ConversationHandler


def main(args):
    if not os.path.exists(args.conf):
        print('Provided configuration file "{}" does not exist! Exiting.'.format(args.conf))
        return
    conf = load_conf(args.conf)

    # setup logging
    if args.logging_level:
        conf['logging_level'] = args.logging_level
    elif 'logging_level' not in conf:
        conf['logging_level'] = 'NOTSET'

    formatter = DialTaskFormatter(
        path_prefix=os.path.join(os.path.abspath(os.path.curdir), 'dialtask/'),
        fmt='%(color)s%(asctime)s [%(levelname)1.1s %(relpath)s:%(lineno)s]%(end_color)s %(message)s',
        datefmt='%H:%M:%S')

    logger = logzero.setup_logger(level=getattr(logging, conf['logging_level']),
                                  formatter=formatter)

    # run the conversation(s)
    handler = ConversationHandler(conf, logger, should_continue=run_for_n_iterations(args.num_dials))
    handler.main_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the DialTask system with a specified configuration file')
    parser.add_argument('--conf', type=str, required=True,
                        help='Path to YAML configuration file with system components definition')
    parser.add_argument('-n', '--num-dials', '--num', type=int, default=1,
                        help='Number of dialogues to run')
    parser.add_argument('-v', '--logging-level', '--verbosity', '--log-level', type=str,
                        choices=['ERROR', 'INFO', 'WARN', 'DEBUG', 'NOTSET'],
                        help='Logging level/verbosity (overriding config defaults)')
    parser.add_argument('-I', '--user-stream-type', '--input-stream-type', '--input-type', type=str,
                        help='Component class to use as input stream (overriding config defaults)')
    parser.add_argument('-O', '--output-stream-type', '--output-type', type=str,
                        help='Component class to use as output stream (overriding config defaults)')
    parser.add_argument('-i', '--input-file', type=str,
                        help='Path to input file (argument to input stream class), if applicable')
    parser.add_argument('-o', '--output-file', type=str,
                        help='Path to output file (argument to output stream class), if applicable')

    args = parser.parse_args()
    main(args)
