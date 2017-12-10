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
                           results_save_dir=self.args.results_dir,
                           seed=self.args.seed,
                           num_envs=self.args.num_envs)

        self.global_step = 0

        self.agent = ACKTRModel(tf.Session(), self.args, self.env.action_space.n)

        # The last state for each env
        self.states = self.env.reset()

    def get_batch(self):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_terminals = []

        # Take the number of steps across all envs to fill a batch

        # TODO: Fix episode summaries with subproc_vec_env
        num_steps = self.args.batch_size // self.args.num_envs
        for step_num in xrange(num_steps):
            # Pick an action and perform it in the envs
            actions = self.agent.get_actions_softmax(self.states)
            next_states, rewards, terminals, infos = self.env.step(actions)

            # TODO: why only saving ep reward for 0th env?
            # This will trigger when the 0th env has a "real done." ie a full episode has finished.
            if infos[0]['real_done']:
                print '-' * 30
                print 'Episode:        ', infos[0]['num_eps']
                print 'Train steps:    ', self.global_step
                print 'Env steps:      ', self.env.num_steps
                print 'Episode reward: ', infos[0]['ep_reward']
                print '-' * 30

                self.agent.write_ep_reward_summary(infos[0]['ep_reward'], infos[0]['env_steps'])


            # Store the SARS
            batch_states.append(self.states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_terminals.append(terminals)

            self.states = next_states

        # Next state for each step in an env is the last state for that env in this batch
        batch_next_states = np.empty((num_steps, self.args.num_envs, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS))
        for i in xrange(num_steps):
            for j in xrange(self.args.num_envs):
                batch_next_states[i, j] = next_states[j]

        # Flipping from num_steps x num_envs to num_envs x num_steps
        #  (20 x 32 to 32 x 20)
        batch_states = np.array(batch_states).swapaxes(1, 0)
        batch_actions = np.array(batch_actions).swapaxes(1, 0)
        batch_next_states = batch_next_states.swapaxes(1, 0)
        batch_rewards = np.array(batch_rewards).swapaxes(1, 0)
        batch_terminals = np.array(batch_terminals).swapaxes(1, 0)


        # Compute the discounted reward
        # NOTE: the discounted reward is computed over the num_steps
        #       rewards earlier get more "look ahead" reward added
        #       to them than later states

        # getting next state for each env, which should be the same for any column
        next_states = batch_next_states[:, 0]
        values_of_next_states = self.agent.value(next_states)
        # looping over envs
        for i, (rewards, terminals) in enumerate(zip(batch_rewards, batch_terminals)):
            # appending value of next state to the rewards if episode hasn't ended
            this_env_terminal = batch_terminals[i][-1]
            if not this_env_terminal:
                v_s_next = values_of_next_states[i]
                rewards = np.append(rewards, v_s_next)
                terminals = np.append(terminals, 0)

            new_rewards = []
            # TODO: They don't stop when they hit a terminal, but maybe we should
            # building up new_rewards in reverse order
            r_d = 0
            for j in reversed(xrange(len(rewards))):
                r = rewards[j]
                t = terminals[j]
                r_d *= self.args.gamma * (1 - t)
                r_d += r
                new_rewards.append(r_d)

            # built new_rewards up in reverse order, so now put it back in time order
            new_rewards.reverse()

            # removing extra entry for value of next state (when not terminal)
            if not this_env_terminal:
                batch_rewards[i, :] = np.array(new_rewards[:-1])
                batch_terminals[i, :] = terminals[:-1]

        return (batch_states.reshape((self.args.batch_size, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS)),
                batch_actions.flatten(),
                batch_rewards.flatten(),
                batch_next_states.reshape((self.args.batch_size, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS)),
                batch_terminals.flatten())


    def run(self):
        print '-' * 30

        while self.env.num_steps < self.args.num_steps * 1.1:
            if self.args.train:
                states, actions, rewards, next_states, terminals = self.get_batch()

                self.global_step = self.agent.train_step(states,
                                                         actions,
                                                         rewards,
                                                         next_states,
                                                         terminals,
                                                         self.env.num_steps)

                print 'Train step %d' % self.global_step

        # Close the env and write monitor results to disk
        self.env.close()

        # The monitor won't be transformed if this script is killed early. In the
        # case that it is, run transform_monitor.py independently.
        transform_monitor(self.args.results_dir, self.args.env)


if __name__ == '__main__':
    runner = Runner(parse_args())
    runner.run()
