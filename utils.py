import gym
import cv2
import argparse
import random
import constants as c

from datetime import datetime
from os.path import join
from sys import maxint

from atari_wrapper import make_atari, wrap_deepmind


#
# CLI
#

def arg_parser():
    """
    Parse input from the command line.

    :return: An ArgumentParser object on which to call parse_args()
    """
    date = date_str()

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--env',
                        help='Gym environment name',
                        default='PongNoFrameskip-v4')
    parser.add_argument('--gpu',
                        help='comma separated list of GPU(s) to use.',
                        default='0')
    parser.add_argument('--seed',
                        help='The random seed.',
                        default=random.randint(0, maxint),
                        type=int)
    parser.add_argument('--render',
                        help='Render the game screen (slows down training).',
                        action='store_true')

    # Training / Evaluation
    parser.add_argument('--no_train',
                        help="Don't train the model while running.",
                        dest='train',
                        action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('--num_eps',
                        help='Max number of episodes to run',
                        default=maxint,
                        type=int)
    parser.add_argument('--num_steps',
                        help='Max number of steps to run',
                        default=maxint,
                        type=int)

    # Paths
    parser.add_argument('--results_dir',
                        help='Output directory for results',
                        default=join('save', 'results', date))
    parser.add_argument('--model_save_path',
                        help='Output directory for models',
                        default=join('save', 'models', date, 'model'))
    parser.add_argument('--model_load_dir',
                        help='Directory of the model you want to load.')
    parser.add_argument('--model_save_name',
                        help='Output directory for models',
                        default=join('save', 'models', date))
    parser.add_argument('--summary_dir',
                        help='Output directory for summaries',
                        default=join('save', 'summaries', date))

    # Hyperparameters
    parser.add_argument('--batch_size',
                        help='Training minibatch size.',
                        default=640,
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate.',
                        default=0.01,
                        type=float)
    parser.add_argument('--damping_lambda',
                        help='The damping factor for _______.', # TODO: Fill in the blank
                        default=0.0,
                        type=float)
    parser.add_argument('--moving_avg_decay',
                        help='The decay factor for the moving average.',
                        default=0.99,
                        type=float)
    parser.add_argument('--kfac_momentum',
                        help='The momentum term for KFAC.',
                        default=0.9,
                        type=float)
    parser.add_argument('--epsilon',
                        help='The epsilon for __________.', # TODO: Fill in the blank
                        default=0.01,
                        type=float)
    parser.add_argument('--drop_rate',
                        help='The dropout rate',
                        default=0.0,
                        type=float)

    return parser


#
# Gym
#

def get_env(env_name, results_save_dir, seed):
    """
    Initialize the OpenAI Gym environment.

    :param env_name: The name of the gym environment to use, (e.g. 'Pong-v0')
    :param results_save_dir: Output directory for results.
    :param seed: The random seed.

    :return: The initialized gym environment.
    """
    env = make_atari(env_name)
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)

    if results_save_dir:
        env = gym.wrappers.Monitor(env, results_save_dir)

    return env


#
# Misc
#

def date_str():
    """
    Gets the current date as a string. Used to create unique directory names.

    :return: A string of the format YYYY-MM-DD.hh:mm:ss'
    """
    # [:-7] cuts off microseconds.
    return str(datetime.now()).replace(' ', '.')[:-7]
