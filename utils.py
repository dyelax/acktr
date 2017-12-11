import gym
import argparse
import random
import csv
import json
from datetime import datetime
from os.path import join, exists
from os import makedirs, environ
from sys import maxint
from glob import glob
from matplotlib import pyplot as plt

from atari_wrapper import make_atari, wrap_deepmind
from subproc_vec_env import SubprocVecEnv
from monitor import Monitor


#
# CLI
#

def parse_args():
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
                        # default='BreakoutNoFrameskip-v4')
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
                        default=10e6,
                        type=int)
    parser.add_argument('--summary_save_freq',
                        help='Frequency to save TensorBoard summaries (in steps)',
                        default=100,
                        type=int)
    parser.add_argument('--model_save_freq',
                        help='Frequency to save the model (in steps',
                        default=100,
                        type=int)

    # Paths
    parser.add_argument('--save_dir',
                        help='Output directory for results',
                        default=join('save', 'results', 'ours', 'pong', date))
                        # default=join('save', 'results', 'ours', 'breakout', date))
    parser.add_argument('--model_load_dir',
                        help='Directory of the model you want to load.')

    # Hyperparameters
    parser.add_argument('--batch_size',
                        help='Training minibatch size.',
                        default=640,
                        type=int)
    parser.add_argument('--num_envs',
                        help='The number of envs to run at once.',
                        default=32,
                        type=int)
    parser.add_argument('--lr',
                        help='Learning rate.',
                        default=0.25,
                        type=float)
    parser.add_argument('--damping_lambda',
                        help='The damping factor for KFAC.',
                        default=0.0,
                        type=float)
    parser.add_argument('--moving_avg_decay',
                        help='The decay factor for the moving average in KFAC.',
                        default=0.99,
                        type=float)
    parser.add_argument('--kfac_momentum',
                        help='The momentum term for KFAC.',
                        default=0.9,
                        type=float)
    parser.add_argument('--drop_rate',
                        help='The dropout rate',
                        default=0.0,
                        type=float)
    parser.add_argument('--k',
                        help='Value of k for k-step lookahead',
                        default=20,
                        type=int)
    parser.add_argument('--gamma',
                        help='The gamma value for k-step lookahead',
                        default=0.99,
                        type=float)

    args = parser.parse_args()

    if args.gpu:
        environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Create save directories if they don't exist
    get_dir(args.save_dir)

    # Multiply num_steps by 1.1 because that's what baselines does for plotting.
    # I think this is so there's always an episode that ends after the intended num_steps so the
    # plot doesn't stop early.
    args.num_steps *= 1.1

    return args


#
# Gym
#

def should_save_vid(ep_i):
    """
    Determines whether to save the episode video.

    :param ep_i: The episode index
    """
    # print 'VID INDEX: ', ep_i
    return  ep_i > 75 and ep_i % 2 == 0


def get_env(env_name, results_save_dir, seed, num_envs):
    """
    Initialize the OpenAI Gym environment.

    :param env_name: The name of the gym environment to use, (e.g. 'Pong-v0')
    :param results_save_dir: Output directory for results.
    :param seed: The random seed.

    :return: The initialized gym environment.
    """

    # Create the 32 environments to parallize
    envs = []
    def make_sub_env_creator(env_num):
        """ Returns a function that creates an event. """
        def sub_env_creator():
            sub_env = make_atari(env_name)
            sub_env.seed(seed + env_num)
            if results_save_dir and env_num == 0:
                sub_env = gym.wrappers.Monitor(sub_env, results_save_dir)
            sub_env = wrap_deepmind(sub_env, frame_stack=True, scale=True)

            return sub_env

        return sub_env_creator

    envs = [make_sub_env_creator(i) for i in range(num_envs)]

    return SubprocVecEnv(envs)


#
# Monitor
#

def transform_monitor(dir, env):
    """
    Transform the new monitor format to be compatible with Baselines monitor
    format used in plot.py and write to disk.

    :param dir: The directory of the monitor files.
    :param env: The gym environment the monitor was tracking.
    """
    json_path = glob(join(dir, '*.stats.json'))[0]
    with open(join(dir, '0.monitor.csv'), 'wb') as csv_file:
        writer = csv.writer(csv_file)
        with open(json_path, 'rb') as json_file:
            blob = json.load(json_file)
            initial_timestamp = blob['initial_reset_timestamp']

            # Do a normal write for the comment line to avoid weird quoting
            comment = '#{"env_id": "%s", "gym_version": "0.9.4", "t_start": %f}\n' \
                       % (env, initial_timestamp)
            csv_file.write(comment)

            header = ['r', 'l', 't']
            writer.writerow(header)

            rows = zip(
                blob['episode_rewards'],
                blob['episode_lengths'],
                [time - initial_timestamp for time in blob['timestamps']]
            )

            writer.writerows(rows)

#
# Misc
#

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not exists(directory):
        makedirs(directory)
    return directory

def date_str():
    """
    Gets the current date as a string. Used to create unique directory names.

    :return: A string of the format YYYY-MM-DD.hh:mm:ss'
    """
    # [:-7] cuts off microseconds.
    return str(datetime.now()).replace(' ', '.')[:-7]


#
#
#

def show_state(s):
    """
    Displays the state for debugging purposes.

    :param s: The state (An array of stacked grayscale images).
    """
    print s.shape
    print s[:,:,0].shape
    print s[:,:,0]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(s[:,:,0], cmap='gray')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(s[:,:,1], cmap='gray')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(s[:,:,2], cmap='gray')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(s[:,:,3], cmap='gray')
    plt.show()
