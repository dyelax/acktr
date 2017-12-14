import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import constants as c
from collections import deque


class Runner:
    def __init__(self, args):
        self.args = args
        self.env = get_env(self.args.env,
                           results_save_dir=self.args.results_save_dir,
                           seed=self.args.seed,
                           num_envs=self.args.num_envs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.agent = ACKTRModel(tf.Session(config=config), self.args, self.env.action_space.n)

        # The last seen state for each env
        self.states = self.env.reset()
        self.terminals = np.repeat([False], self.args.num_envs)
        self.global_step = 0

        # The buffer size is 2 * num_steps
        self.num_steps = self.args.batch_size // self.args.num_envs
        buff_size = self.num_steps * 2
        self.batch_states_buff = deque(maxlen=buff_size)
        self.batch_actions_buff = deque(maxlen=buff_size)
        self.batch_rewards_buff = deque(maxlen=buff_size)
        self.batch_terminals_buff = deque(maxlen=buff_size)

        # Prefill the buffer with 20 steps
        self.collect_steps()

    def collect_steps(self):
        # Take the number of steps across all envs to fill a batch
        for step_num in xrange(self.num_steps):
            # Pick an action and perform it in the envs
            actions = self.agent.get_actions(self.states)

            # Store the SARS
            # states, actions and terminals are appended before env.step and rewards after because
            # everything should be relative to the state (state was terminal and we took action
            # from it and that lead to getting reward).
            # TODO: These copies might not be necessary.
            self.batch_states_buff.append(np.copy(self.states))
            self.batch_terminals_buff.append(np.copy(self.terminals))
            self.batch_actions_buff.append(actions)

            self.states, rewards, self.terminals, infos = self.env.step(actions)

            self.batch_rewards_buff.append(rewards)

            # This will trigger when the 0th env has finished a full episode.
            if infos[0]['real_done']:
                print '-' * 30
                print 'Episode:        ', infos[0]['num_eps']
                print 'Train steps:    ', self.global_step
                print 'Env steps:      ', self.env.num_steps
                print 'Episode reward: ', infos[0]['ep_reward']
                print '-' * 30

                self.agent.write_ep_reward_summary(infos[0]['ep_reward'], infos[0]['env_steps'])

        self.batch_terminals_buff.append(self.terminals)

    def get_batch(self):
        # Collect an additional 20 env_steps
        self.collect_steps()

        # Flipping from num_steps x num_envs to num_envs x num_steps
        #  (40 x 32 to 32 x 40)
        batch_states = np.array(self.batch_states_buff).swapaxes(1, 0)
        batch_actions = np.array(self.batch_actions_buff).swapaxes(1, 0)
        batch_rewards = np.array(self.batch_rewards_buff).swapaxes(1, 0)
        batch_terminals = np.array(self.batch_terminals_buff).swapaxes(1, 0)

        batch_terminals = batch_terminals[:, 1:]

        # Loop over envs and Compute the discounted reward
        for i, (env_rewards, env_terminals) in enumerate(zip(batch_rewards, batch_terminals)):
            # The value of the next_states for each step in the current env
            values_of_next_states = self.agent.get_values(batch_states[self.num_steps:])
            discounted_rewards = []

            # For each step in the env
            for step in self.num_steps:
                r_d = 0
                discount = 1

                # Discount over the 20 next states
                for j in self.num_steps:
                    reward = env_rewards[step + j]
                    terminal = env_terminals[step + j]

                    if terminal: # Stop discounting when we reach a terminal
                        r_d += reward
                        break
                    else:
                        r_d += reward * discount

                    discount *= self.args.gamma

                # Add value of next state if the last state num steps ahead is not terminal
                if not batch_terminals[j + self.num_steps] and j == self.num_steps - 1:
                    r_d += values_of_next_states[step]

                discounted_rewards.append(r_d)

        # Reshape into (batch_size,) + element.shape
        batch_states = batch_states[:self.num_steps].reshape((-1, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS))
        batch_actions = batch_actions[:self.num_steps].flatten()
        batch_rewards = np.array(discounted_rewards).flatten() # discounted_rewards is already 20 long

        return batch_states, batch_actions, batch_rewards


    def run(self):
        print '-' * 30

        # TODO: Figure out why q_runner is so important
        coord = tf.train.Coordinator()
        enqueue_threads = self.agent.q_runner.create_threads(self.agent.sess, coord=coord, start=True)

        while self.env.num_steps < self.args.num_steps:
            states, actions, rewards = self.get_batch()

            if self.args.train:
                self.agent.train_step(states, actions, rewards, self.env.num_steps)
                self.global_step += 1

            print 'Train step %d' % self.global_step

        coord.request_stop()
        coord.join(enqueue_threads)

        # Close the env and write monitor results to disk
        self.env.close()

        # The monitor won't be transformed if this script is killed early. In the
        # case that it is, run transform_monitor.py independently.
        transform_monitor(self.args.results_save_dir, self.args.env)


if __name__ == '__main__':
    runner = Runner(parse_args())
    runner.run()
