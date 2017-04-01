from __future__ import print_function
import argparse

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-seed", default=0, type=int)
    parser.add_argument("--dynet-mem", default=512, type=int)
    parser.add_argument("--dynet-gpus", default=0, type=int)
    parser.add_argument("--test_out", '-teo',
                        default='results/test.en-de.de', type=str)
    parser.add_argument("--model_in", '-min', type=str, help='Model to load from')
    parser.add_argument("--model_out", '-mout', type=str, help='Model to save to')
    parser.add_argument('--num_iters', '-ni',
                        type=int, help='Number of iters', default=10000)
    parser.add_argument('--batch_size', '-bs',
                        type=int, help='minibatch size', default=100)
    parser.add_argument('--dev_batch_size', '-dbs', default=200,
                        type=int, help='minibatch size for the validation set')
    parser.add_argument('--learning_rate', '-lr',
                        type=float, help='learning rate', default=0.1)
    parser.add_argument('--learning_rate_decay', '-lrd',
                        type=float, help='learning rate decay', default=0.01)
    parser.add_argument('--valid_every', '-ve',
                        type=int, help='Check valid error every', default=100)
    parser.add_argument('--test_every', '-te',
                        type=int, help='Run on test set every', default=500)
    parser.add_argument("--shuffle",
                        help="Shuffle training set",
                        action="store_true")
    parser.add_argument("--verbose", '-v',
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--exp_name", '-en', type=str, required=True,
                        help='Name of the experiment (used so save the model)')
    return parser

def mlp_parser():
    parser = base_parser()
    parser.add_argument('--hidden_dim', '-dh',
                        type=str, help='Dimensions of the hidden layers with the format "100" or "100-100", etc..', default='800-800')
    parser.add_argument('--activation', '-a',
                        type=str, help='Activation for the hidden layers', default='relu')
    return parser

def rnn_parser():
    parser = base_parser()
    parser.add_argument('--hidden_dim', '-dh',
                        type=int, help='hidden size', default=250)
    parser.add_argument('--num_samples', '-ns',
                        type=int, help='Number of example to sample at each checkpoint', default=5)
    parser.add_argument("--char_level", '-cl',
                        help="Run on character level (vs word level)",
                        action="store_true")
    return parser

def print_config(args):
    print('======= CONFIG =======')
    for k, v in vars(args).items():
        print(k, ':', v)
    print('======================')