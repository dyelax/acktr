import tensorflow as tf
import numpy as np

from utils import get_env, parse_args, transform_monitor, show_state

from random_agent import RandomAgent
from acktr_model import ACKTRModel
import collections
import constants as c
import kfac

from atari_wrapper import EpisodicLifeEnv

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def mse(pred, target):
    return tf.square(pred - target) / 2.


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


class Scheduler(object):
    def __init__(self, v, nvalues):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

    def value(self):
        current_value = self.v * (1 - (self.n / self.nvalues))
        self.n += 1.
        return current_value


class Model(object):

    def __init__(self, args, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear'):
        self.args = args
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        self.agent = ACKTRModel(self.sess, self.args, nact)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.agent.pi, labels=A)
        self.logits = logits = self.agent.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(cat_entropy(self.agent.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(self.agent.vf), R))
        train_loss = pg_loss + vf_coef * vf_loss


        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = self.agent.vf + tf.random_normal(tf.shape(self.agent.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(self.agent.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params=params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss,params)

        # with tf.device('/gpu:0'):
        self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,
            momentum=0.9, kfac_update=1, epsilon=0.01,
            stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=max_grad_norm)

        update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
        train_op, q_runner, global_step_op = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps)

        def train(obs, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {self.agent.x_batch:obs, A:actions, ADV:advs, R:rewards, PG_LR:cur_lr}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.get_actions = self.agent.get_actions
        self.get_values = self.agent.get_values
        self.write_ep_reward_summary = self.agent.write_ep_reward_summary
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps, nstack, gamma):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

        self.global_step = 0

    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        for n in range(self.nsteps):
            actions = self.model.get_actions(self.obs)
            values = self.model.get_values(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)

            # This will trigger when the 0th env has a "real done." ie a full episode has finished.
            if infos[0]['real_done']:
                print '-' * 30
                print 'Episode:        ', infos[0]['num_eps']
                print 'Train steps:    ', self.global_step
                print 'Env steps:      ', self.env.num_steps
                print 'Episode reward: ', infos[0]['ep_reward']
                print '-' * 30

                self.model.write_ep_reward_summary(infos[0]['ep_reward'], infos[0]['env_steps'])

        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape((-1, 84, 84, 4))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.get_values(self.obs).tolist()
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
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values

def learn(args, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
                 nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear'):
    tf.reset_default_graph()

    env = get_env(args.env, args.results_dir, args.seed, args.num_envs)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda : Model(args, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule)
    model = make_model()

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    nbatch = nenvs*nsteps
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)

    for update in range(1, total_timesteps//nbatch+1):
        obs, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, rewards, masks, actions, values)
        model.old_obs = obs

        runner.global_step += 1
        # print 'Train Step', runner.global_step

    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()



if __name__ == '__main__':
    # runner = Runner(parse_args())
    # runner.run()

    learn(parse_args())
