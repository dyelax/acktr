import gym
import cv2
import argparse
import random

from datetime import datetime
from os.path import join
from sys import maxint

from atari_wrapper import FireResetEnv, MapState, FrameStack, LimitLength

#
# CLI
#

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parser():
    """
    Parse input from the command line.

    :return: An ArgumentParser object on which to call parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        help='Gym environment name',
                        default='Pong-v0')
    parser.add_argument('--gpu',
                        help='comma separated list of GPU(s) to use.',
                        default='0')
    parser.add_argument('--load',
                        help='Path to the model you want to load.')
    parser.add_argument('--results_save_dir',
                        help='Output directory for results',
                        default=join('save', 'results', date_str()))
    parser.add_argument('--model_save_dir',
                        help='Output directory for model',
                        default=join('save', 'models', date_str()))
    parser.add_argument('--num_eps',
                        help='Max number of episodes to run',
                        default=maxint,
                        type=int)
    parser.add_argument('--num_steps',
                        help='Max number of steps to run',
                        default=maxint,
                        type=int)
    parser.add_argument('--seed',
                        help='The random seed.',
                        default=random.randint(0, maxint),
                        type=int)
    parser.add_argument('--render',
                        help='Render the game screen (slows down training).',
                        action='store_true')
    parser.add_argument('--evaluate',
                        help='Evaluate instead of training',
                        action='store_true')

    return parser

#
# Data
#

def resize(img, img_size):
    return cv2.resize(img, img_size)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def preprocess(img):
    img = resize(img, (84, 84))
    img = grayscale(img)
    return img

#
# Gym
#

def get_env(env_name, evaluate, results_save_dir, seed):
    """
    Initialize the OpenAI Gym environment.

    :param env_name: The name of the gym environment to use, (e.g. 'Pong-v0')
    :param evaluate: Whether the environment is for evaluation instead of training.
    :param results_save_dir: Output directory for results.
    :param seed: The random seed.

    :return: The initialized gym environment.
    """
    env = gym.make(env_name)
    env.seed(seed)

    env = FireResetEnv(env)
    env = MapState(env, preprocess)
    # TODO (Mike): frame skipping / max over 2 consecutive frames.
    # Described here: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    env = FrameStack(env, 4)

    if not evaluate: env = LimitLength(env, 60000)
    if results_save_dir: env = gym.wrappers.Monitor(env, results_save_dir)

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
