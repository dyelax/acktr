import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import collections
import constants as c

from atari_wrapper import EpisodicLifeEnv

class Runner:
    def __init__(self, args):
        self.args = args
        self.env = get_env(self.args.env,
                           results_save_dir=self.args.save_dir,
                           seed=self.args.seed,
                           num_envs=self.args.num_envs)

        self.global_step = 0

        self.agent = ACKTRModel(tf.Session(), self.args, self.env.action_space.n)

        # The last state for each env
        self.states = self.env.reset()
        self.terminals = np.repeat([False], self.args.num_envs)

    def get_batch(self):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_terminals = []

        # Take the number of steps across all envs to fill a batch
        num_steps = self.args.batch_size // self.args.num_envs
        for step_num in xrange(num_steps):
            # Pick an action and perform it in the envs
            actions = self.agent.get_actions(self.states)

            # Store the SARS
            # states, actions and terminals are appended before env.step and rewards after because
            # everything should be relative to the state (state was terminal and we took action
            # from it and that lead to getting reward).
            # TODO: These copies might not be necessary.
            batch_states.append(np.copy(self.states))
            batch_terminals.append(np.copy(self.terminals))
            batch_actions.append(actions)

            self.states, rewards, self.terminals, infos = self.env.step(actions)

            batch_rewards.append(rewards)

            # This will trigger when the 0th env has finished a full episode.
            if infos[0]['real_done']:
                print '-' * 30
                print 'Episode:        ', infos[0]['num_eps']
                print 'Train steps:    ', self.global_step
                print 'Env steps:      ', self.env.num_steps
                print 'Episode reward: ', infos[0]['ep_reward']
                print '-' * 30

                self.agent.write_ep_reward_summary(infos[0]['ep_reward'], infos[0]['env_steps'])

        batch_terminals.append(self.terminals)

        # Flipping from num_steps x num_envs to num_envs x num_steps
        #  (20 x 32 to 32 x 20)
        batch_states = np.array(batch_states).swapaxes(1, 0)
        batch_actions = np.array(batch_actions).swapaxes(1, 0)
        batch_rewards = np.array(batch_rewards).swapaxes(1, 0)
        batch_terminals = np.array(batch_terminals).swapaxes(1, 0)

        batch_terminals = batch_terminals[:, 1:]

        # Compute the discounted reward
        # NOTE: the discounted reward is computed over the num_steps
        #       rewards earlier get more "look ahead" reward added
        #       to them than later states

        # # The value of the next_state for each env
        values_of_next_states = self.agent.get_values(self.states)
        # # Loop over envs
        # for i, (env_rewards, env_terminals) in enumerate(zip(batch_rewards, batch_terminals)):
        #     # Append value of next state to the rewards if episode didn't end on the last step
        #     env_last_terminal = batch_terminals[i, -1]
        #     if not env_last_terminal:
        #         v_s_next = values_of_next_states[i]
        #         env_rewards = np.append(env_rewards, v_s_next)
        #         env_terminals = np.append(env_terminals, 0)
        #
        #     new_rewards = []
        #     # TODO: They don't stop when they hit a terminal, but maybe we should
        #     # building up new_rewards in reverse order
        #     discounted_future_reward = 0
        #     for j in reversed(xrange(len(env_rewards))):
        #         reward = env_rewards[j]
        #         terminal = env_terminals[j]
        #         discounted_future_reward *= self.args.gamma * (1 - terminal)
        #         discounted_future_reward += reward
        #         new_rewards.append(discounted_future_reward)
        #
        #     # built new_rewards up in reverse order, so now put it back in time order
        #     new_rewards.reverse()
        #
        #     # removing extra entry for value of next state (when not terminal)
        #     if not env_last_terminal:
        #         batch_rewards[i, :] = np.array(new_rewards[:-1])
        #         batch_terminals[i, :] = env_terminals[:-1]

        # TODO: Try swapping in their discount
        def discount_with_dones(rewards, dones, gamma):
            discounted = []
            r = 0
            for reward, done in zip(rewards[::-1], dones[::-1]):
                r = reward + gamma * r * (1. - done)  # fixed off by one bug
                discounted.append(r)
            return discounted[::-1]

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(batch_rewards, batch_terminals, values_of_next_states)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.args.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.args.gamma)
            batch_rewards[n] = rewards

        # Reshape into (batch_size,) + element.shape
        batch_states = batch_states.reshape((-1, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS))
        batch_actions = batch_actions.flatten()
        batch_rewards = batch_rewards.flatten()

        return batch_states, batch_actions, batch_rewards


    def run(self):
        print '-' * 30

        # TODO: Do we need q_runner here?
        while self.env.num_steps < self.args.num_steps:
            self.global_step += 1
            if self.args.train:
                states, actions, rewards = self.get_batch()
                self.agent.train_step(states, actions, rewards)

                print 'Train step %d' % self.global_step

        # Close the env and write monitor results to disk
        self.env.close()

        # The monitor won't be transformed if this script is killed early. In the
        # case that it is, run transform_monitor.py independently.
        transform_monitor(self.args.save_dir, self.args.env)


if __name__ == '__main__':
    runner = Runner(parse_args())
    runner.run()
