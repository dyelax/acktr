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
        self.defineGraph()

        self.saver = tf.train.Saver()

        checkPoint = tf.train.get_checkpoint_state(self.args.model_load_dir)
        modelExists = checkPoint and checkPoint.model_checkpoint_path
        if modelExists:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        #set up new writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)


    def registerFisherBlock(self, conv_block, layer_key, **kwargs):
        if conv_block:
            fisher_block = tf.contrib.kfac.fisher_blocks.ConvKFCBasicFB(layer_collection=self.layer_collection, 
                **kwargs)
        else:
            fisher_block = tf.contrib.kfac.fisher_blocks.FullyConnectedKFACBasicFB(layer_collection=self.layer_collection, 
                **kwargs)
        self.layer_collection.register_block(layer_key=layer_key, fisher_block=fisher_block)


    # TODO: entropy regularization
    def defineGraph(self):
        self.glob_step = tf.Variable(0, name="global_step", trainable=False)
        self.x_Batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=[None])
        self.r_labels = tf.placeholder(dtype=tf.float32, shape=[None])

        self.layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()

        #convs
        channel_sizes = [c.IN_CHANNELS].extend(c.CHANNEL_SIZES)
        prevLayer = self.x_Batch
        for i in xrange(1, c.NUM_CONV_LAYERS):
            in_channels, out_channels, kernelSize, stride = channel_sizes[i - 1], channel_sizes[i], \
                c.CONV_KERNEL_SIZES[i], c.CONV_STRIDES[i]
            w_shape = kernelSize.extend([in_channels, out_channels])
            w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.01))
            b = tf.Variable(tf.truncated_normal(shape=[out_channels], stddev=0.01))

            curLayer = tf.nn.conv2d(prevLayer, filter=w, strides=stride, padding="SAME") + b # TODO: padding?
            self.registerFisherBlock(conv_block=True, layer_key=("conv_%d" % i), params=(w, b), inputs=prevLayer, 
                outputs=curLayer, strides=stride, padding="SAME")
            curLayer = tf.nn.elu(curLayer)
            prevLayer = curLayer

        #fully connected layer (last shared)
        conv_shape = curLayer.shape
        flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
        flattened = tf.reshape(curLayer, shape=[-1, flat_sz])

        fc_layer = tf.layers.dense(flattened, c.FC_SIZE, activation=None)
        self.registerFisherBlock(conv_block=False, layer_key="fc_layer", inputs=flattened, 
            outputs=fc_layer, has_bias=True)
        fc_layer = tf.nn.elu(fc_layer)

        #policy output layer
        policy_fc_layer = tf.layers.dense(fc_layer, self.num_actions)
        self.registerFisherBlock(conv_block=False, layer_key="policy_fc_layer", inputs=fc_layer, 
            outputs=policy_fc_layer, has_bias=True)
        self.policy_probs = tf.nn.softmax(policy_fc_layer)

        #value output layer
        self.value_preds = tf.layers.dense(fc_layer, 1)
        self.registerFisherBlock(conv_block=False, layer_key="value_fc_layer", inputs=fc_layer, 
            outputs=self.value_preds, has_bias=True)
        probs_of_actions_taken = tf.reduce_sum(self.policy_probs * tf.one_hot(self.actions_taken, 
            depth=self.num_actions), axis=1)
        
        self.actor_loss = -np.reduce_sum(np.log(probs_of_actions_taken) * self.r_labels) # TODO: is this multiplication right? need to dot instead?
        self.critic_loss = 0.5 * tf.losses.mean_squared_error(self.r_labels, self.value_preds)

        optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(self.args.lr,
            cov_ema_decay=self.args.moving_avg_decay, damping=self.args.damping_lambda,
            layer_collection=self.layer_collection, momentum=self.args.kfac_momentum)

        #only passing global step to one train op so it isn't double-incremented
        self.train_op_actor = optimizer.minimize(self.actor_loss, global_step=self.global_step)
        self.train_op_critic = optimizer.minimize(self.critic_loss)
        
        #summaries
        self.A_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
        self.C_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)


    def train_step(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        k = self.args.k #the k from k-step return
        V_S = sess.run([self.value_preds], feed_dict={self.x_Batch: s_batch})
        V_S_next = sess.run([self.value_preds], feed_dict={self.x_Batch: s_next_batch})
        V_S_next *= terminal_batch #mask out preds for termainl states
        label_batch = (r_batch + V_S_next * np.exp(self.args.gamma, self.args.k)) - V_S

        #train critic
        sessArgs = [self.global_step, self.C_loss_summary, self.train_op_critic]
        feedDict = {self.x_Batch: s_batch, 
                    self.r_labels: label_batch}
        step, C_summary, _ = self.sess.run(sessArgs, feed_dict=feedDict)

        #train actor
        sessArgs = [self.A_loss_summary, self.train_op_actor]
        feedDict = {self.x_Batch: s_batch, 
                    self.r_labels: label_batch
                    self.action_taken: a_batch}
        A_summary, _ = self.sess.run(sessArgs, feed_dict=feedDict)

        if step % c.SAVE_FREQ == 0:
            self.summary_writer.add_summary(C_summary, global_step = step))
            self.summary_writer.add_summary(A_summary, global_step = step))
            self.saver.save(self.sess, self.args.model_save_path, global_step = step)


    # TODO later: increase temp of softmax over time?
    def predict(self):
        '''
        Predict all Q values for a state -> softmax dist -> sample from dist
        '''
        return np.random.choice(len(self.policy_probs), 1, p=self.policy_probs)[0]


