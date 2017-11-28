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

        self.saver = tf.train.Saver()

        model_exists = False
        if self.args.model_load_dir:
            check_point = tf.train.get_checkpoint_state(self.args.model_load_dir)
            model_exists = check_point and check_point.model_checkpoint_path
        if model_exists:
            self.saver.restore(self.sess, check_point.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)


    def fully_connected_layer(self, inputs, input_size, output_size, name='fc_layer'):
        w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], stddev=0.01), name=("%s/W" % name))
        b = tf.Variable(tf.truncated_normal(shape=[output_size], stddev=0.01), name=("%s/b" % name))
        outputs = tf.matmul(inputs, w) + b
        self.layer_collection.register_fully_connected(params=(w,b), inputs=inputs, outputs=outputs)
        return outputs


    def define_graph(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.x_batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=[None])
        self.r_labels = tf.placeholder(dtype=tf.float32, shape=[None])

        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        #convs
        channel_sizes = [c.IN_CHANNELS] + c.CHANNEL_SIZES
        prev_layer = self.x_batch
        for i in xrange(c.NUM_CONV_LAYERS):
            in_channels, out_channels = channel_sizes[i], channel_sizes[i+1]
            kernel_size, stride = c.CONV_KERNEL_SIZES[i], (1,) + c.CONV_STRIDES[i] + (1,)
            w_shape = kernel_size + (in_channels, out_channels)
            w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.01), name=("conv_%d/W" % i))
            b = tf.Variable(tf.truncated_normal(shape=[out_channels], stddev=0.01), name=("conv_%d/b" % i))

            cur_layer = tf.nn.conv2d(prev_layer, filter=w, strides=stride, padding="SAME") + b # TODO: padding?
            self.layer_collection.register_conv2d(params=(w, b), inputs=prev_layer, 
                outputs=cur_layer, strides=stride, padding="SAME")
            cur_layer = tf.nn.elu(cur_layer)
            prev_layer = cur_layer

        #fully connected layer (last shared)
        conv_shape = cur_layer.shape
        flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
        flattened = tf.reshape(cur_layer, shape=[-1, flat_sz])

        fc_layer = self.fully_connected_layer(flattened, c.FC_INPUT_SIZE, c.FC_SIZE, 'fc_layer')
        fc_layer = tf.nn.elu(fc_layer)

        #policy output layer
        policy_fc_layer = self.fully_connected_layer(fc_layer, c.FC_SIZE, self.num_actions, 'policy_fc_layer')
        self.policy_probs = tf.nn.softmax(policy_fc_layer)

        #value output layer
        self.value_preds = self.fully_connected_layer(fc_layer, c.FC_SIZE, 1, 'value_fc_layer')
        self.value_preds = tf.squeeze(self.value_preds)
        probs_of_actions_taken = tf.reduce_sum(self.policy_probs * tf.one_hot(self.actions_taken, 
            depth=self.num_actions), axis=1)

        self.layer_collection.register_categorical_predictive_distribution(self.policy_probs, seed=self.args.seed)
        self.layer_collection.register_normal_predictive_distribution(self.value_preds, var=1, seed=self.args.seed)

        self.actor_loss = -tf.reduce_sum(tf.log(probs_of_actions_taken) * self.r_labels)
        self.critic_loss = 0.5 * tf.losses.mean_squared_error(self.r_labels, self.value_preds)
        self.entropy_regularization = -tf.reduce_sum(self.policy_probs * tf.log(self.policy_probs))

        self.total_loss = self.actor_loss + self.critic_loss + c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy_regularization

        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(self.args.lr,
            cov_ema_decay=self.args.moving_avg_decay, damping=self.args.damping_lambda,
            layer_collection=self.layer_collection, momentum=self.args.kfac_momentum)

        self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
        
        #summaries
        self.a_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.c_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)

        self.ep_reward = tf.placeholder(tf.float32)
        self.ep_reward_summary = tf.summary.scalar("episode_reward", self.ep_reward)


    def train_step(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        k = self.args.k #the k from k-step return
        v_s = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_batch})
        v_s_next = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_next_batch})
        v_s_next *= terminal_batch #mask out preds for termainl states
        label_batch = (r_batch + v_s_next * (self.args.gamma ** self.args.k)) - v_s
        label_batch = np.squeeze(label_batch.transpose())

        sess_args = [self.global_step, self.a_loss_summary, self.c_loss_summary, self.train_op]
        feed_dict = {self.x_batch: s_batch, 
                    self.r_labels: label_batch,
                    self.actions_taken: a_batch}
        step, a_summary, c_summary, _ = self.sess.run(sess_args, feed_dict=feed_dict)

        if step % c.SAVE_FREQ == 0:
            self.summary_writer.add_summary(a_summary, global_step=step)
            self.summary_writer.add_summary(c_summary, global_step=step)
            self.saver.save(self.sess, self.args.model_save_path, global_step=step)

        return step


    # TODO later: increase temp of softmax over time?
    def get_action(self, state):
        '''
        Predict all Q values for a state -> softmax dist -> sample from dist
        '''
        feed_dict = {self.x_batch: state}
        policy_probs = self.sess.run(self.policy_probs, feed_dict=feed_dict)
        policy_probs = np.squeeze(policy_probs)
        return np.random.choice(len(policy_probs), p=policy_probs)


    def write_ep_reward_summary(self, ep_reward, steps):
        summary = self.sess.run(self.ep_reward_summary,
                                feed_dict={self.ep_reward: ep_reward})

        self.summary_writer.add_summary(summary, global_step=steps)


