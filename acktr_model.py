import kfac, kfac_utils
#from baselines_utils import find_trainable_variables, ortho_init
from baselines_utils import ortho_init
import constants as c
import glob
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# noinspection PyAttributeOutsideInit
class ACKTRModel:
    def __init__(self, sess, args, num_actions):
        self.sess = sess
        self.args = args
        self.num_actions = num_actions
        self.define_graph()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        if self.args.model_load_dir:
            check_point = tf.train.get_checkpoint_state(self.args.model_load_dir)
            model_exists = check_point and check_point.model_checkpoint_path
            if model_exists:
                print 'Restoring model from ' + check_point.model_checkpoint_path
                self.saver.restore(self.sess, check_point.model_checkpoint_path)

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(self.args.save_dir, self.sess.graph)



    def define_graph(self):
        def conv(x, scope, nf, rf, stride, pad='VALID', act=tf.nn.relu, init_scale=1.0):
            with tf.variable_scope(scope):
                nin = x.get_shape()[3].value
                w = tf.get_variable("w", [rf, rf, nin, nf], initializer=ortho_init(init_scale))
                b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0.0))
                z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
                h = act(z)
                return h

        def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
            with tf.variable_scope(scope):
                nin = x.get_shape()[1].value
                w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
                b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
                z = tf.matmul(x, w) + b
                h = act(z)
                return h

        def sample(logits):
            noise = tf.random_uniform(tf.shape(logits))
            return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        def conv_to_fc(x):
            nh = np.prod([v.value for v in x.get_shape()[1:]])
            x = tf.reshape(x, [-1, nh])
            return x

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

    #        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.x_batch = tf.placeholder(dtype=tf.uint8, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32)
        self.actor_labels = tf.placeholder(dtype=tf.float32)
        self.critic_labels = tf.placeholder(dtype=tf.float32)

#        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        with tf.variable_scope("model", reuse=False):
            # noinspection PyTypeChecker
            h = conv(tf.cast(self.x_batch, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', self.num_actions, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        self.value_preds = vf[:, 0]
        self.action_preds = sample(pi)

        self.A = tf.placeholder(tf.int32)
        self.ADV = tf.placeholder(tf.float32)
        self.R = tf.placeholder(tf.float32)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=self.A)
        self.logits = logits = pi

        ##training loss
        pg_loss = tf.reduce_mean(self.ADV * logpac)
        entropy = tf.reduce_mean(cat_entropy(pi))
        self.policy_loss = pg_loss - 0.01 * entropy
        self.value_loss = tf.reduce_mean(mse(tf.squeeze(vf), self.R))
        train_loss = pg_loss + 0.5 * self.value_loss

        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = vf + tf.random_normal(tf.shape(vf))
        self.vf_fisher = vf_fisher_loss = - 1.0 * tf.reduce_mean(
            tf.pow(vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params = params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)



        # with tf.device('/gpu:0'):
        self.optim = optim = kfac.KfacOptimizer(learning_rate=self.learning_rate, clip_kl=0.001,
                                                momentum=0.9, kfac_update=1, epsilon=0.01,
                                                stats_decay=0.99, async=1, cold_iter=10,
                                                max_grad_norm=0.5)

        update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
        # print params
        # print grads
        self.train_op, self.q_runner, self.global_step_op = optim.apply_gradients(list(zip(grads, params)))
        self.lr = Scheduler(v=self.args.lr, nvalues=self.args.num_steps)

        #summaries
        # self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        # self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)

        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)

    def calculate_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)


    def get_values(self, s_batch):
        v_s = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_batch})
        return np.squeeze(v_s)


    def train_step(self, s_batch, a_batch, r_batch):
        # percent_done = float(env_steps) / self.args.num_steps

        v_s = self.get_values(s_batch)
#        v_s_next = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_next_batch})
#        v_s_next *= (1 - terminal_batch) #mask out preds for terminal states

        #create labels
#        k_step_return = (r_batch + v_s_next * (self.args.gamma ** (self.args.k + 1))) #estimated k-step return
        k_step_return = r_batch
        advantage = k_step_return - v_s #estimated k-step return - v_s
        #reshape to remove extra dim
        k_step_return = np.reshape(k_step_return, [-1]) #turn into row vec
        advantage = np.reshape(advantage, [-1]) #turn into row vec

        for step in range(len(s_batch)):
            cur_lr = self.lr.value()

        sess_args = self.train_op
        feed_dict = {self.x_batch: s_batch,
                     self.actor_labels: advantage,
                     self.critic_labels: k_step_return,
                     self.actions_taken: a_batch,
                     self.learning_rate: cur_lr,
                     self.R: r_batch,
                     self.ADV: advantage,
                     self.A: a_batch}
        self.sess.run(sess_args, feed_dict=feed_dict)

        # if (step - 1) % self.args.summary_save_freq == 0:
        #     self.summary_writer.add_summary(a_summary, global_step=step)
        #     self.summary_writer.add_summary(c_summary, global_step=step)
        #
        # if (step - 1) % self.args.model_save_freq == 0:
        #     self.saver.save(self.sess, self.args.model_save_path, global_step=step)
        #
        # return step


    # TODO later: increase temp of softmax over time?
    # def get_actions_softmax(self, states):
    #     '''
    #     Predict all Q values for a state -> softmax dist -> sample from dist
    #     '''
    #     feed_dict = {self.x_batch: states}
    #     policy_probs = self.sess.run(self.policy_probs, feed_dict=feed_dict)
    #     # policy_probs = np.squeeze(policy_probs)
    #     actions = np.array([np.random.choice(len(state_probs), p=state_probs) for state_probs in policy_probs])
    #     return actions

    def get_actions(self, states):
        feed_dict = {self.x_batch: states}
        actions = self.sess.run(self.action_preds, feed_dict=feed_dict)
        return actions

    def write_ep_reward_summary(self, ep_reward, steps):
        summary = self.sess.run(self.ep_reward_summary,
                                feed_dict={self.ep_reward: ep_reward})

        self.summary_writer.add_summary(summary, global_step=steps)

from utils import parse_args

if __name__ == '__main__':
    num_actions = 5
    sess = tf.Session()
    args = parse_args()
    model = ACKTRModel(sess, args, num_actions)
    batch_size = 10
    for _ in xrange(100):
        model.train_step(np.random.rand(batch_size,c.IN_WIDTH,c.IN_HEIGHT,c.IN_CHANNELS),
                        np.random.randint(num_actions, size=batch_size),
                        np.random.rand(batch_size),
                        np.random.rand(batch_size,c.IN_WIDTH,c.IN_HEIGHT,c.IN_CHANNELS),
                        np.random.randint(2, size=batch_size),
                        1)
        print(model.get_actions(np.random.rand(1,84,84,4)))
