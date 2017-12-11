import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import collections
import constants as c

from atari_wrapper import EpisodicLifeEnv


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class Runner(object):

    def __init__(self, env, agent, nsteps, nstack, gamma):
        self.env = env
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.agent = agent

        self.global_step = 0

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        for n in range(self.nsteps):
            actions = self.agent.get_actions(self.obs)
            values = self.agent.get_values(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            # self.obs = obs
            mb_rewards.append(rewards)

            # This will trigger when the 0th env has a "real done." ie a full episode has finished.
            if infos[0]['real_done']:
                print '-' * 30
                print 'Episode:        ', infos[0]['num_eps']
                print 'Train steps:    ', self.global_step
                print 'Env steps:      ', self.env.num_steps
                print 'Episode reward: ', infos[0]['ep_reward']
                print '-' * 30

                self.agent.write_ep_reward_summary(infos[0]['ep_reward'], infos[0]['env_steps'])

        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape((-1, 84, 84, 4))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.agent.get_values(self.obs).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        return mb_obs, mb_rewards, mb_actions

def learn(args):
    tf.reset_default_graph()

    env = get_env(args.env,
                  results_save_dir=args.save_dir,
                  seed=args.seed,
                  num_envs=args.num_envs)

    agent = ACKTRModel(tf.Session(), args, env.action_space.n)

    runner = Runner(env, agent, nsteps=20, nstack=4, gamma=0.99)
    coord = tf.train.Coordinator()
    enqueue_threads = agent.q_runner.create_threads(agent.sess, coord=coord, start=True)
    for update in xrange(int(args.num_steps // args.batch_size)):
        obs, rewards, actions = runner.run()
        agent.train_step(obs, actions, rewards)

        runner.global_step += 1
        print 'Train step %d' % runner.global_step

    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()



if __name__ == '__main__':
    # runner = Runner(parse_args())
    # runner.run()

    learn(parse_args())
