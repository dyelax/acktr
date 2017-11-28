import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import collections
import constants as c


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

def run(args):
    global batch

    env = get_env(args.env,
                  results_save_dir=args.results_dir,
                  seed=args.seed)

    sess = tf.Session()
    agent = ACKTRModel(sess, args, env.action_space.n)
    # agent = RandomAgent(sess, args, env.action_space.n)
    sess.run(tf.global_variables_initializer())

    env_steps = 0
    global_step = 0

    print '-' * 30
    for ep in xrange(args.num_eps):
        print 'Episode: ', ep

        LOOK_AHEAD_BUFF_SIZE = LOOK_AHEAD_BUFF_SIZE + 1

        look_ahead_buff = collections.deque(maxlen=LOOK_AHEAD_BUFF_SIZE)
        reset_batch()

        state = env.reset()
        terminal = False
        ep_reward = 0

        while True:
            # Fill up the batch until it is full or we reach a terminal state
            if len(batch['action']) < args.batch_size:
                start_state = state
                action = agent.get_action(np.expand_dims(state, axis=0))
                state, reward, terminal, _ = env.step(action)

                 # The SARS queue is full so the first item will be popped off
                if len(look_ahead_buff) == LOOK_AHEAD_BUFF_SIZE:
                    popped_sars = look_ahead_buff[0]

                    # Compute the discounted reward
                    r_d = 0
                    for i in xrange(LOOK_AHEAD_BUFF_SIZE):
                        r_d += look_ahead_buff[i][2] * args.gamma**i

                    # Add the SARS to the batch
                    print popped_sars[0], popped_sars[1], r_d, look_ahead_buff[-1][0]
                    add_sars_to_batch(popped_sars[0], popped_sars[1], r_d, look_ahead_buff[-1][0])

                    # Add the state to the look_ahead_buff
                    look_ahead_buff.append((start_state, action, reward))

                    if terminal:
                        for i in xrange(LOOK_AHEAD_BUFF_SIZE):
                            for j in xrange(i, LOOK_AHEAD_BUFF_SIZE - i):
                                r_d += look_ahead_buff[j][2] * args.gamma**j
                            add_sars_to_batch(look_ahead_buff[j][0], look_ahead_buff[j[1], r_d, NULL_STATE, terminal=True)
                        look_ahead_buff = collections.deque(maxlen=LOOK_AHEAD_BUFF_SIZE)
                else:
                    # Add the state to the look_ahead_buff
                    look_ahead_buff.append((start_state, action, reward))

                ep_reward += reward
            else:
                if args.render:
                    env.render()

                if args.train:
                    # Convert the batch dict to numpy arrays
                    states = np.array(batch['state'])
                    actions = np.array(batch['action'])
                    rewards = np.array(batch['reward'])
                    next_states = np.array(batch['next_state'])
                    terminals = np.array(batch['terminal'])

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

                    # Reset the batch
                    reset_batch()

                if terminal or env_steps > args.num_steps:
                    break

            env_steps += 1

        agent.write_ep_reward_summary(ep_reward, env_steps)
        print 'Train steps:    ', global_step
        print 'Env steps:      ', env_steps
        print 'Episode reward: ', ep_reward
        print '-' * 30

        if env_steps > args.num_steps:
            break

    # Close the env and write monitor results to disk
    env.close()

    # The monitor won't be transformed if this script is killed early. In the
    # case that it is, run transform_monitor.py independently.
    transform_monitor(args.results_dir, args.env)


if __name__ == '__main__':
    run(parse_args())
