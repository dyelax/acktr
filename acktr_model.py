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
        w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], stddev=0.01, seed=self.args.seed), name=("%s/W" % name))
        b = tf.Variable(tf.truncated_normal(shape=[output_size], stddev=0.01, seed=self.args.seed), name=("%s/b" % name))
        outputs = tf.matmul(inputs, w) + b
        self.layer_collection.register_fully_connected(params=(w,b), inputs=inputs, outputs=outputs)
        return outputs


    def define_graph(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.x_batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=[None])
        self.actor_labels = tf.placeholder(dtype=tf.float32, shape=[None])
        self.critic_labels = tf.placeholder(dtype=tf.float32, shape=[None])

        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        #convs
        channel_sizes = [c.IN_CHANNELS] + c.CHANNEL_SIZES
        prev_layer = self.x_batch
        for i in xrange(c.NUM_CONV_LAYERS):
            in_channels, out_channels = channel_sizes[i], channel_sizes[i+1]
            kernel_size, stride = c.CONV_KERNEL_SIZES[i], (1,) + c.CONV_STRIDES[i] + (1,)
            w_shape = kernel_size + (in_channels, out_channels)
            w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.01, seed=self.args.seed), name=("conv_%d/W" % i))
            b = tf.Variable(tf.truncated_normal(shape=[out_channels], stddev=0.01, seed=self.args.seed), name=("conv_%d/b" % i))

            cur_layer = tf.nn.conv2d(prev_layer, filter=w, strides=stride, padding="VALID") + b # TODO: padding?
            self.layer_collection.register_conv2d(params=(w, b), inputs=prev_layer, 
                outputs=cur_layer, strides=stride, padding="VALID")
            cur_layer = tf.nn.relu(cur_layer)
            prev_layer = cur_layer

        #fully connected layer (last shared)
        conv_shape = cur_layer.shape
        flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
        flattened = tf.reshape(cur_layer, shape=[-1, flat_sz])

        fc_layer = self.fully_connected_layer(flattened, flat_sz, c.FC_SIZE, 'fc_layer')
        fc_layer = tf.nn.relu(fc_layer)

        #policy output layer
        self.policy_logits = self.fully_connected_layer(fc_layer, c.FC_SIZE, self.num_actions, 'policy_logits')
        self.policy_probs = tf.nn.softmax(self.policy_logits)

        #value output layer
        self.value_preds = self.fully_connected_layer(fc_layer, c.FC_SIZE, 1, 'value_fc_layer')
        self.value_preds = tf.squeeze(self.value_preds)
        
        #done with sparse soft max now

        self.layer_collection.register_categorical_predictive_distribution(self.policy_logits, seed=self.args.seed)
        self.layer_collection.register_normal_predictive_distribution(self.value_preds, var=1, seed=self.args.seed)

        #done with sparse soft max now
            #probs_of_actions_taken = tf.reduce_sum(self.policy_probs * tf.one_hot(self.actions_taken, depth=self.num_actions), axis=1)
            #self.actor_loss = -tf.reduce_mean(tf.log(probs_of_actions_taken) * self.actor_labels)
        self.actor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits, labels=self.actions_taken) * self.actor_labels)
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_labels - self.value_preds)) / 2.0
        
        #TODO: implement more stable version that they do in code
        self.entropy_regularization = tf.reduce_mean(self.calculate_entropy(self.policy_logits))
        self.actor_loss = self.actor_loss - c.ENTROPY_REGULARIZATION_WEIGHT * self.entropy_regularization

        self.total_loss = self.actor_loss + 0.5 * self.critic_loss

        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(self.learning_rate,
            cov_ema_decay=self.args.moving_avg_decay, damping=self.args.damping_lambda,
            layer_collection=self.layer_collection, momentum=self.args.kfac_momentum)

        self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
        
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


    def train_step(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch, env_steps):
        percent_done = float(env_steps) / self.args.num_steps

        v_s = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_batch})
        v_s_next = self.sess.run([self.value_preds], feed_dict={self.x_batch: s_next_batch})
        v_s_next *= terminal_batch #mask out preds for terminal states
        
        #create labels
        critic_return_labels = (r_batch + v_s_next * (self.args.gamma ** self.args.k)) #estiated k-step return
        actor_advantage_labels = critic_return_labels - v_s #estiated k-step return - v_s
        #reshape to remove extra dim
        critic_return_labels = np.reshape(critic_return_labels, [-1]) #turn into row vec
        actor_advantage_labels = np.reshape(actor_advantage_labels, [-1]) #turn into row vec

        sess_args = [self.global_step, self.a_loss_summary, self.c_loss_summary, self.train_op]
        feed_dict = {self.x_batch: s_batch, 
                    self.actor_labels: actor_advantage_labels,
                    self.critic_labels: critic_return_labels,
                    self.actions_taken: a_batch,
                    self.learning_rate: self.args.lr * (1 - percent_done)}
        step, a_summary, c_summary, _ = self.sess.run(sess_args, feed_dict=feed_dict)

        if (step - 1) % self.args.summary_save_freq == 0:
            self.summary_writer.add_summary(a_summary, global_step=step)
            self.summary_writer.add_summary(c_summary, global_step=step)

        if (step - 1) % self.args.model_save_freq == 0:
            self.saver.save(self.sess, self.args.model_save_path, global_step=step)

        return step


    # TODO later: increase temp of softmax over time?
    def get_action_softmax(self, state):
        '''
        Predict all Q values for a state -> softmax dist -> sample from dist
        '''
        feed_dict = {self.x_batch: state}
        policy_probs = self.sess.run(self.policy_probs, feed_dict=feed_dict)
        policy_probs = np.squeeze(policy_probs)
        return np.random.choice(len(policy_probs), p=policy_probs)

    def get_action(self, state):
        feed_dict = {self.x_batch: state}
        policy_logits = self.sess.run(self.policy_logits, feed_dict=feed_dict)
        policy_logits = np.squeeze(policy_logits)
        noise = np.random.rand(*policy_logits.shape)
        return np.argmax(policy_logits - np.log(-np.log(noise)))

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
    model.train_step(np.random.rand(batch_size,c.IN_WIDTH,c.IN_HEIGHT,c.IN_CHANNELS),
                        np.random.randint(num_actions, size=batch_size),
                        np.random.rand(batch_size),
                        np.random.rand(batch_size,c.IN_WIDTH,c.IN_HEIGHT,c.IN_CHANNELS),
                        np.random.randint(2, size=batch_size),
                        1)
    print(model.get_action(np.random.rand(1,84,84,4)))
