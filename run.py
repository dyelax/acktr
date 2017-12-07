import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import collections
import constants as c

from atari_wrapper import EpisodicLifeEnv


NULL_STATE = np.zeros((c.IN_WIDTH, c.IN_HEIGHT, c.IN_CHANNELS))

batch = {}

def add_sars_to_batch(state, action, r_d, next_state, terminal=False):
    global batch
    batch['state'].append(state)
    batch['action'].append(action)
    batch['reward'].append(r_d)
    batch['terminal'].append(terminal)
    batch['next_state'].append(next_state)

def reset_batch():
    global batch
    batch = {
        'state': [],
        'action': [],
        'reward': [],
        'next_state': [],
        'terminal': []
    }

def get_episodic_life_env(env):
    try:
        while not isinstance(env, EpisodicLifeEnv):
            env = env.env
    except AttributeError:
        'No episodic life wrapper for env.'

    return env


def get_batch(agent):
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    # Take the number of steps across all envs to fill a batch

    # TODO: Fix episode summaries with subproc_vec_env
    # num_steps = args.batch_size // args.num_envs
    # for step_num in xrange(num_steps):
    #     # Pick an action and perform it in the envs
    #     actions = agent.get_action(states)
    #     next_states, rewards, terminals, _ = env.step(actions)
    #
    #     ep_reward += rewards[0]
    #
    #     if get_episodic_life_env(env).was_real_done_last_reset:
    #         print 'Terminal'
    #         print '-' * 30
    #         print 'Episode:        ', num_eps
    #         print 'Train steps:    ', global_step
    #         print 'Env steps:      ', env_steps
    #         print 'Episode reward: ', ep_reward
    #         print '-' * 30
    #
    #         agent.write_ep_reward_summary(ep_reward, env_steps)
    #
    #         num_eps += 1
    #         ep_reward = 0
    #     else:
    #         print 'Death'
    #
    #     if (terminals[0]):
    #         ep_reward = 0


        # Store the SARS
        batch_states.append(states)
        batch_actions.append(actions)
        batch_rewards.append(rewards)
        batch_terminals.append(terminals)

        states = next_states

    # Next state for each step in an env is the last state for that env in this batch
    batch_next_states = np.empty((args.num_envs, num_steps))
    for i in xrange(num_steps):
        for j in xrange(args.num_envs):
            batch_next_states[i, j] = next_states[j]

    # Flipping from num_steps x num_envs to num_envs x num_steps
    #  (20 x 32 to 32 x 20)
    batch_states = np.array(batch_states).swapaxes(1, 0)
    batch_actions = np.array(batch_actions).swapaxes(1, 0)
    batch_next_states = np.array(batch_next_states).swapaxes(1, 0)
    batch_rewards = np.array(batch_rewards).swapaxes(1, 0)
    batch_terminals = np.array(batch_terminals).swapaxes(1, 0)


    # Compute the discounted reward
    # NOTE: the discounted reward is computed over the num_steps
    #   rewards earlier get more "look ahead" reward added to them than later states
    for i, rewards in enumerate(batch_rewards):
        new_rewards = []
        r_d = 0
        # TODO: They don't stop when they hit a terminal, but maybe we should
        for j, r in enumerate(rewards):
            r_d = r * args.gamma ** j
            new_rewards.append(r_d)

        batch_rewards[i, :] = np.array(new_rewards)

    return (batch_states.flatten(),
            batch_actions.flatten(),
            batch_rewards.flatten(),
            batch_next_states.flatten(),
            batch_terminals.flatten())

    # Fill up the batch
    # if len(batch['action']) < args.batch_size:
    #     start_state = state
    #     action = agent.get_action(np.expand_dims(state, axis=0))
    #     # action = agent.get_action_softmax(np.expand_dims(state, axis=0))
    #     state, reward, terminal, _ = env.step(action)
    #     ep_reward += reward
    #
    #      # The SARS queue is full so the first item will be popped off
    #     if len(look_ahead_buff) == LOOK_AHEAD_BUFF_SIZE:
    #         popped_sars = look_ahead_buff[0]
    #
    #         # Compute the discounted reward
    #         r_d = 0
    #         for i in xrange(LOOK_AHEAD_BUFF_SIZE):
    #             r_d += look_ahead_buff[i][2] * args.gamma**i
    #
    #         # Add the SARS to the batch
    #         # print popped_sars[1], r_d
    #         add_sars_to_batch(popped_sars[0], popped_sars[1], r_d, look_ahead_buff[-1][0])
    #
    #         # Add the state to the look_ahead_buff
    #         look_ahead_buff.append((start_state, action, reward))
    #
    #         if terminal:
    #             for i in xrange(LOOK_AHEAD_BUFF_SIZE):
    #                 r_d = 0
    #                 for j in xrange(LOOK_AHEAD_BUFF_SIZE - i):
    #                     r_d += look_ahead_buff[i + j][2] * args.gamma**j
    #
    #                 add_sars_to_batch(look_ahead_buff[j][0],
    #                                   look_ahead_buff[j][1],
    #                                   r_d,
    #                                   NULL_STATE,
    #                                   terminal=True)
    #     else:
    #         # Add the state to the look_ahead_buff
    #         look_ahead_buff.append((start_state, action, reward))
    #
    # else:


def run(args):
    global batch

    env = get_env(args.env,
                  results_save_dir=args.results_dir,
                  seed=args.seed,
                  num_envs=args.num_envs)

    print
    sess = tf.Session()
    agent = ACKTRModel(sess, args, env.action_space.n)
    # agent = RandomAgent(sess, args, env.action_space.n)

    num_eps = 0
    env_steps = 0
    global_step = 0
    ep_reward = 0
    reset_batch()

    print '-' * 30

    if args.train:
        # Convert the batch dict to numpy arrays
        states, actions, rewards, next_states, terminals = get_batch(agent)

        # Visualize 3 random states from the batch
        # for i in xrange(3):
        #     i = np.random.choice(len(states))
        #     show_state(states[i])

        global_step = agent.train_step(states,
                                       actions,
                                       rewards,
                                       next_states,
                                       terminals,
                                       env_steps)

        if global_step % 10 == 0:
            print 'Train step ', global_step

        # Reset the batch
        reset_batch()

        if terminal:
            break

        env_steps += 1

    if env_steps > args.num_steps * 1.1:
        break

    # Close the env and write monitor results to disk
    env.close()

    # The monitor won't be transformed if this script is killed early. In the
    # case that it is, run transform_monitor.py independently.
    transform_monitor(args.results_dir, args.env)


if __name__ == '__main__':
    run(parse_args())
