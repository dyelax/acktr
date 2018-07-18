import kfac, kfac_utils
#from baselines_utils import find_trainable_variables, ortho_init
from baselines_utils import ortho_init
import constants as c
import glob
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class ACKTRModel:
    def __init__(self, sess, args, num_actions):
        self.sess = sess
        self.args = args
        self.num_actions = num_actions
        self.define_graph()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        model_exists = False
        if self.args.model_load_dir:
            check_point = tf.train.get_checkpoint_state(self.args.model_load_dir)
            model_exists = check_point and check_point.model_checkpoint_path
            if model_exists:
                print 'Restoring model from ' + check_point.model_checkpoint_path
                self.saver.restore(self.sess, check_point.model_checkpoint_path)

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_save_dir, self.sess.graph)


    def fully_connected_layer(self, inputs, output_size, name='fc_layer', init_scale=1.0, activation=None):
#         w = tf.get_variable("%s/W" % name, [input_size, output_size], initializer=ortho_init(init_scale))
#         b = tf.get_variable("%s/b" % name, [output_size], initializer=tf.constant_initializer(0.0))
# #        w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], stddev=0.01, seed=self.args.seed), name=("%s/W" % name))
# #        b = tf.Variable(tf.truncated_normal(shape=[output_size], stddev=0.01, seed=self.args.seed), name=("%s/b" % name))
#         outputs = tf.matmul(inputs, w) + b
# #        self.layer_collection.register_fully_connected(params=(w,b), inputs=inputs, outputs=outputs)
#         return outputs

        functional_layer = tf.layers.Dense(
            output_size,
            kernel_initializer=ortho_init(init_scale),
            bias_initializer=tf.constant_initializer(0.0),
            name=name)
        pre_activations=functional_layer(inputs)
        self.layer_collection.register_fully_connected(
            inputs=inputs,
            outputs=pre_activations,
            params=(functional_layer.kernel, functional_layer.bias))
        return activation(pre_activations) if activation else pre_activations

    def conv2d(self, inputs, kernel_size, stride, output_channels, name='conv_layer'):
        functional_layer = tf.layers.Conv2D(
                    output_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    kernel_initializer=ortho_init(np.sqrt(2)),
                    bias_initializer=tf.constant_initializer(0.0),
                    padding="VALID",
                    activation=None,
                    name=name)
        pre_activations=functional_layer(inputs)
        self.layer_collection.register_conv2d(
            inputs=inputs,
            outputs=pre_activations,
            params=(functional_layer.kernel, functional_layer.bias), 
            strides=(1,)+stride+(1,), 
            padding="VALID")
        return tf.nn.relu(pre_activations)

    def define_graph(self):
        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

#        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.x_batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32)
        self.actor_labels = tf.placeholder(dtype=tf.float32)
        self.critic_labels = tf.placeholder(dtype=tf.float32)

        self.global_step = tf.Variable(0, name="step", trainable=False)

#        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        with tf.variable_scope("model", reuse=False):

            #convs
            channel_sizes = [c.IN_CHANNELS] + c.CHANNEL_SIZES
            #why???
            prev_layer = tf.cast(self.x_batch, tf.float32)
            for i in xrange(c.NUM_CONV_LAYERS):
                in_channels, out_channels = channel_sizes[i], channel_sizes[i+1]
                #kernel_size, stride = c.CONV_KERNEL_SIZES[i], (1,) + c.CONV_STRIDES[i] + (1,)
                kernel_size, stride = c.CONV_KERNEL_SIZES[i], c.CONV_STRIDES[i]


                cur_layer = self.conv2d(prev_layer, kernel_size, stride, out_channels, name="conv_%d/W" % i)
    #             w_shape = kernel_size + (in_channels, out_channels)
    #             w = tf.get_variable("conv_%d/W" % i, w_shape, initializer=ortho_init(np.sqrt(2)))
    #             b = tf.get_variable("conv_%d/b" % i, [out_channels], initializer=tf.constant_initializer(0.0))
    # #            w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.01, seed=self.args.seed), name=("conv_%d/W" % i), )
    # #            b = tf.Variable(tf.truncated_normal(shape=[out_channels], stddev=0.01, seed=self.args.seed), name=("conv_%d/b" % i))

    #             cur_layer = tf.nn.conv2d(prev_layer, filter=w, strides=stride, padding="VALID") + b
    # #            self.layer_collection.register_conv2d(params=(w, b), inputs=prev_layer,
    # #                outputs=cur_layer, strides=stride, padding="VALID")
    #             cur_layer = tf.nn.relu(cur_layer)
                


                prev_layer = cur_layer

            #fully connected layer (last shared)
            conv_shape = cur_layer.shape
            flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
            flattened = tf.reshape(cur_layer, shape=[-1, flat_sz])

            #tanh?
            fc_layer = self.fully_connected_layer(flattened, c.FC_SIZE, 'fc_layer', init_scale=np.sqrt(2), activation=tf.nn.relu)

            #policy output layer
            self.policy_logits = self.fully_connected_layer(fc_layer, self.num_actions, 'policy_logits')
            self.policy_probs = tf.nn.softmax(self.policy_logits)

            #value output layer
            self.value_preds = self.fully_connected_layer(fc_layer, 1, 'value_fc_layer')
            self.value_preds = tf.squeeze(self.value_preds, axis=1)

            #seed?
            self.layer_collection.register_categorical_predictive_distribution(self.policy_logits)
            self.layer_collection.register_normal_predictive_distribution(self.value_preds, var=1.0)
    #        self.layer_collection.register_categorical_predictive_distribution(self.policy_logits, seed=self.args.seed)
    #        self.layer_collection.register_normal_predictive_distribution(self.value_preds, var=1, seed=self.args.seed)

            params = tf.trainable_variables() #"model" scope's variables



        #intentionally defined outside of scope.. Loss calcluations:

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits, labels=self.actions_taken)

        self.actor_loss = tf.reduce_mean(self.actor_labels * logpac)
        self.critic_loss = tf.reduce_mean(tf.square(self.value_preds - self.critic_labels)) / 2.0

        self.entropy_regularization = tf.reduce_mean(self.calculate_entropy(self.policy_logits))
        self.actor_loss = self.actor_loss - c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy_regularization

        self.total_loss = self.actor_loss + 0.5 * self.critic_loss

        # pg_fisher_loss = -tf.reduce_mean(logpac)
        # sample_net = self.value_preds + tf.random_normal(tf.shape(self.value_preds))
        # vf_fisher_loss = -1.0 * tf.reduce_mean(tf.pow(self.value_preds - tf.stop_gradient(sample_net), 2))
        # joint_fisher_loss = pg_fisher_loss + vf_fisher_loss


        #Gradients and updates:

        #params = find_trainable_variables("model")
        grads = tf.gradients(self.total_loss, params)
        grad_params = list(zip(grads, params))

        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(layer_collection=self.layer_collection, damping=self.args.damping_lambda,
                                                            learning_rate=self.learning_rate, cov_ema_decay=self.args.moving_avg_decay,
                                                            momentum=self.args.kfac_momentum, norm_constraint=self.args.kfac_norm_constraint)

        cov_variable_thunks, cov_update_thunks, inv_variable_thunks, inv_update_thunks = optimizer.create_ops_and_vars_thunks()

        self.cov_variable_ops = [thunk() for thunk in cov_variable_thunks]
        self.cov_update_ops = [thunk() for thunk in cov_update_thunks]
        self.cov_update_op = tf.group(*self.cov_update_ops)
        self.inv_variable_ops = [thunk() for thunk in inv_variable_thunks]
        # inv_update_ops = [thunk() for thunk in inv_update_thunks]
        
        estimator = optimizer._fisher_est
        self.inv_update_ops =  []
        for factor in estimator._layers.get_factors():
            # with estimator._inv_device_context_generator():
            for op in factor.make_inverse_update_ops():
                self.inv_update_ops.append(op)

        self.train_op = optimizer.apply_gradients(grad_params)


        # q = tf.contrib.kfac.op_queue.OpQueue(self.inv_update_ops)
        # self.q_runner = tf.train.QueueRunner(q, [q.next_op(self.sess)])

        # queue = tf.contrib.kfac.op_queue
        # enqueue_op = queue.enqueue(self.inv_update_ops)
        # self.q_runner = tf.train.QueueRunner(queue, [enqueue_op])

        # self.train_op = optimizer.apply_gradients(grad_params)

        # self.global_step_op = tf.assign(global_step, global_step+1)
        
        # self.cov_update_op = optimizer.cov_update_op
        # nec? 
        # self.inv_update_op = optimizer.inv_update_op
        # for q_runner in train! queue.dequeue()
        # inv_update_dict = optimizer.inv_updates_dict
        # nec? 
        # self.factors = self.layer_collection.get_factors()

        # print self.inv_update_ops[0].dtype
        # print inv_update_ops
        # exit()

        #make q_runner and self.dequeue_op (called qr in the original kfac.py optimizer that we are reimplementing)
        factorOps_dummy = [i for i in self.inv_update_ops if not type(i) is tf.Operation] #list(inv_update_dict.values()) is the dict values the same as the ops???
        # dtypes = [factorOps_dummy[0].dtype for i in xrange(len(factorOps_dummy))]
        # print dtypes
        # for i in xrange(len(factorOps_dummy)):
        #     print "----"
        #     print i
        #     print factorOps_dummy[i]
        #     if not type(factorOps_dummy[i]) is tf.Operation:
        #         print factorOps_dummy[i].dtype
        #     # print "*"
        #     # print inv_update_ops[i]
        #     # if inv_update_ops[i].OpDef != tf.no_op:
        #     #     print inv_update_ops[i].dtype
        #     # print "*"
        queue = tf.FIFOQueue(1, [item.dtype for item in factorOps_dummy], #should be tf.float32???
                             shapes=[item.get_shape() for item in factorOps_dummy])
        enqueue_op = queue.enqueue(factorOps_dummy)
        #note, the above is instead of the following lines on the original kfac implementation:
            # enqueue_op = tf.cond(tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update), tf.convert_to_tensor(
            #     0)), tf.greater_equal(self.stats_step, self._stats_accum_iter)), lambda: queue.enqueue(self.computeStatsEigen()), tf.no_op)
        self.dequeue_op = queue.dequeue()
        self.q_runner = tf.train.QueueRunner(queue, [enqueue_op])




        self.global_step_op = tf.assign(self.global_step, self.global_step+1)


        #lr correct?


        # self.optim = optim = kfac.KfacOptimizer(learning_rate=self.learning_rate, clip_kl=0.001,
        #             momentum=0.9, kfac_update=1, epsilon=0.01,
        #             stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=0.5)


#        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(self.learning_rate,
#            cov_ema_decay=self.args.moving_avg_decay, damping=self.args.damping_lambda,
#            layer_collection=self.layer_collection, momentum=self.args.kfac_momentum)

#        self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

        # TODO: is this return value necessary?
        # update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
        # self.train_op, self.q_runner, self.global_step_op = optim.apply_gradients(list(zip(grads,params)))

        #summaries
        self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)

        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)

    def calculate_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)


    def get_values(self, s_batch):
        v_s = self.sess.run(self.value_preds, feed_dict={self.x_batch: s_batch})
        return v_s


    def train_step(self, s_batch, a_batch, r_batch, env_steps):
        percent_done = float(env_steps) / (1.1 * self.args.num_steps)

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

        sess_args = [self.global_step_op, self.a_loss_summary, self.c_loss_summary, self.train_op, self.cov_update_op]
        feed_dict = {self.x_batch: s_batch,
                    self.actor_labels: advantage,
                    self.critic_labels: k_step_return,
                    self.actions_taken: a_batch,
                    self.learning_rate: self.args.lr * (1 - percent_done)}
        step, a_summary, c_summary, _, _ = self.sess.run(sess_args, feed_dict=feed_dict)

        if (step - 1) % self.args.summary_save_freq == 0:
            self.summary_writer.add_summary(a_summary, global_step=step)
            self.summary_writer.add_summary(c_summary, global_step=step)

        if (step - 1) % self.args.model_save_freq == 0:
            self.saver.save(self.sess, os.path.join(self.args.model_save_dir, 'model'), global_step=step)

        return step

    def get_actions(self, states):
        """
        Predict all Q values for a state -> softmax dist -> sample from dist

        :param states: A batch of states from the environment.

        :return: A list of the action for each state
        """
        # TODO: Increase temp of softmax over time?
        feed_dict = {self.x_batch: states}
        policy_probs = self.sess.run(self.policy_probs, feed_dict=feed_dict)
        # policy_probs = np.squeeze(policy_probs)
        actions = np.array([np.random.choice(len(state_probs), p=state_probs) for state_probs in policy_probs])
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
                        1)
        print(model.get_actions(np.random.rand(1,84,84,4)))
