from baselines_utils import ortho_init
import constants as c
import glob
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class ACKTRModel:
    """
    Class used for ACKTR model. Contains a value network and a policy network, both trained used the KFAC optimizer.
    """
    def __init__(self, sess, args, num_actions):
        """
        The initializer for the ACKTRModel

        :param sess: the tf.Session instance to be associated with the graph
        :param args: the args from argparse
        :param num_actions: the number of discrete actions over the policy network can act 
                            (i.e. the size of its output)
        """
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
        """
        Returns a tensor representing the relevant dense on the input. 
        Also registers the layer in a layer_collection for KFAC.

        :param inputs: the tensor on which the transformation will be done
        :param output_size: the number of neurons in the output layer
        :param name: the name to be given to the layer
        :param init_scale: the scale passed to ortho_init
        :param activation: the activation to be used on the dense layer. (None is deafult.)

        :return: a tensor representing the activations
        """
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
        """
        Returns a tensor representing the relevant convolution on the input. 
        Also registers the layer in a layer_collection for KFAC.

        :param inputs: the tensor on which the convolution will be done
        :param kernel_size: the size of the kernel to be used
        :param stride: the stride in each dimension for the kernel
        :param output_channels: the number of dimensions in the next later / number of filters to use
        :param name: the name to be given to the layer

        :return: a tensor representing the activations
        """
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
        """
        Creates all the tensors for the policy and value networks and creates update ops.
        """
        # inputs #
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.x_batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32)
        self.actor_labels = tf.placeholder(dtype=tf.float32)
        self.critic_labels = tf.placeholder(dtype=tf.float32)
        #step
        self.global_step = tf.Variable(0, name="step", trainable=False)
        #layer collection for KFAC optimizer
        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        # model layers and parameters #
        with tf.variable_scope("model", reuse=False):
            last_shared_fc_layer = self.create_shared_layers()
            #policy output layer
            self.policy_logits = self.fully_connected_layer(last_shared_fc_layer, self.num_actions, 'policy_logits')
            self.policy_probs = tf.nn.softmax(self.policy_logits)
            #value output layer
            self.value_preds = self.fully_connected_layer(last_shared_fc_layer, 1, 'value_fc_layer')
            self.value_preds = tf.squeeze(self.value_preds, axis=1)
            #register final layers
            self.layer_collection.register_categorical_predictive_distribution(self.policy_logits) #can put seed here for debugging
            self.layer_collection.register_normal_predictive_distribution(self.value_preds, var=1.0)
            #get trainable vars
            params = tf.trainable_variables() #"model" scope's variables

        # loss calcluations #
        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits, labels=self.actions_taken)
        self.actor_loss = tf.reduce_mean(self.actor_labels * logpac)
        self.critic_loss = tf.reduce_mean(tf.square(self.value_preds - self.critic_labels)) / 2.0
        self.entropy_regularization = tf.reduce_mean(self.calculate_entropy(self.policy_logits))
        self.actor_loss = self.actor_loss - c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy_regularization
        self.total_loss = self.actor_loss + 0.5 * self.critic_loss

        # training ops #
        grads = tf.gradients(self.total_loss, params)
        grad_params = list(zip(grads, params))
        self.optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(layer_collection=self.layer_collection, damping=self.args.damping_lambda,
                                                            learning_rate=self.learning_rate, cov_ema_decay=self.args.moving_avg_decay,
                                                            momentum=self.args.kfac_momentum, norm_constraint=self.args.kfac_norm_constraint)
        self.train_op = self.optimizer.apply_gradients(grad_params)

        # q_runner needed for parallelism.
        self.create_q_runner_and_cov_op()
        #summaries
        self.create_summaries()
        #global step
        self.global_step_op = tf.assign(self.global_step, self.global_step+1)

    def create_shared_layers(self):
        """
        Creates the layers shared between the value net and the policy net. (convolional layers and 1 dense layer)
        """
        #convs
        channel_sizes = [c.IN_CHANNELS] + c.CHANNEL_SIZES
        prev_layer = tf.cast(self.x_batch, tf.float32)
        for i in xrange(c.NUM_CONV_LAYERS):
            in_channels, out_channels = channel_sizes[i], channel_sizes[i+1]
            kernel_size, stride = c.CONV_KERNEL_SIZES[i], c.CONV_STRIDES[i]
            cur_layer = self.conv2d(prev_layer, kernel_size, stride, out_channels, name="conv_%d/W" % i)
            prev_layer = cur_layer
        #fully connected layer (last shared)
        conv_shape = cur_layer.shape
        flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
        flattened = tf.reshape(cur_layer, shape=[-1, flat_sz])
        #tanh activation in some implementations I beleive
        return self.fully_connected_layer(flattened, c.FC_SIZE, 'last_shared_fc_layer', init_scale=np.sqrt(2), activation=tf.nn.relu)

    def create_summaries(self):
        """
        Creates summarries to be run later by sess.run
        """
        self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)
        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)

    def create_q_runner_and_cov_op(self):
        """
        Creates q_runner used by run.py in order to mult-process. Also create the covariance update operation.
        """
        # found how do to do these few lines in https://github.com/gd-zhang/ACKTR/blob/master/models/model.py
        self.cov_update_op = self.optimizer.cov_update_op
        # inv_update_op = self.optimizer.inv_update_op #this op is unnecessary given q_runner as far as I can tell
        inv_update_dict = self.optimizer.inv_updates_dict
        factors = self.layer_collection.get_factors()
        inv_updates = list(inv_update_dict.values())
        queue = tf.FIFOQueue(1, [item.dtype for item in inv_updates],
                                [item.get_shape() for item in inv_updates])
        # in baseline ACKTR, was this
        # enqueue_op = tf.cond(tf.equal(tf.mod(self.global_step_tensor, self.inv_iter), tf.convert_to_tensor(0)),
        #                      lambda: queue.enqueue(self.model.inv_update_dict.value()), tf.no_op)
        # but now we can do this
        enqueue_op = queue.enqueue(list(inv_updates))
        self.dequeue_op = queue.dequeue()
        self.q_runner = tf.train.QueueRunner(queue, [enqueue_op])


    def calculate_entropy(self, logits):
        """
        Calculate the entropy of the logits

        :param logits: the logits of the network (neurons pre-softmax)
        """
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=1)


    def get_values(self, s_batch):
        """
        Use value net on a batch of states

        :param s_batch: the input states
        """
        v_s = self.sess.run(self.value_preds, feed_dict={self.x_batch: s_batch})
        return v_s


    def train_step(self, s_batch, a_batch, r_batch, env_steps):
        """
        Complete one trianing step

        :param s_batch: the input states
        :param a_batch: the actions take
        :param r_batch: the k-step return
        :param env_steps: the number of steps in environment
        """
        percent_done = float(env_steps) / (1.1 * self.args.num_steps)

        v_s = self.get_values(s_batch)

        #create labels
        k_step_return = r_batch
        advantage = k_step_return - v_s #estimated k-step return - v_s
        #reshape to remove extra dim
        k_step_return = np.reshape(k_step_return, [-1]) #turn into row vec
        advantage = np.reshape(advantage, [-1]) #turn into row vec

        sess_args = [self.global_step_op, self.a_loss_summary, self.c_loss_summary, self.train_op, self.cov_update_op, self.dequeue_op]
        feed_dict = {self.x_batch: s_batch,
                    self.actor_labels: advantage,
                    self.critic_labels: k_step_return,
                    self.actions_taken: a_batch,
                    self.learning_rate: self.args.lr * (1 - percent_done)}
        step, a_summary, c_summary, _, _, _ = self.sess.run(sess_args, feed_dict=feed_dict)

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
        """
        Writes a summary to disk for the episode

        :param ep_reward: the reward for this episode
        :param steps: the global step
        """
        summary = self.sess.run(self.ep_reward_summary,
                                feed_dict={self.ep_reward: ep_reward})

        self.summary_writer.add_summary(summary, global_step=steps)



#for a quick functionality validation, we can run this file with the following example:
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
