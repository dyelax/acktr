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


    def defineGraph(self):
        self.glob_step = tf.Variable(0, name="global_step", trainable=False)
        self.x_Batch = tf.placeholder(dtype=tf.float32, shape=[None, c.IN_HEIGHT, c.IN_WIDTH, c.IN_CHANNELS])
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=[None])
        self.r_d = tf.placeholder(dtype=tf.float32, shape=[None])

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

            fisher_block = tf.contrib.kfac.fisher_blocks.ConvKFCBasicFB(layer_collection=self.layer_collection, 
                params=(w, b), inputs=prevLayer, outputs=curLayer, strides=stride, padding="SAME")
            self.layer_collection.register_block(layer_key="conv_i", fisher_block=fisher_block)

            curLayer = tf.nn.elu(curLayer)
            prevLayer = curLayer

        #fully connected
        conv_shape = curLayer.shape
        flat_sz = conv_shape[1].value * conv_shape[2].value * conv_shape[3].value
        flattened = tf.reshape(curLayer, shape=[-1, flat_sz]) 
        fc_layer = tf.layers.dense(flattened, c.FC_SIZE, activation=None)

        fisher_block = tf.contrib.kfac.fisher_blocks.FullyConnectedKFACBasicFB(layer_collection=self.layer_collection, 
            inputs=flattened, outputs=fc_layer, has_bias=True)
        self.layer_collection.register_block(layer_key="fc_layer", fisher_block=fisher_block)

        fc_layer = tf.nn.elu(fc_layer)

        #policy output layer
        policy_fc_layer = tf.layers.dense(fc_layer, self.num_actions)

        fisher_block = tf.contrib.kfac.fisher_blocks.FullyConnectedKFACBasicFB(layer_collection=self.layer_collection, 
            inputs=fc_layer, outputs=policy_fc_layer, has_bias=True)
        self.layer_collection.register_block(layer_key="policy_fc_layer", fisher_block=fisher_block)
        
        self.policy_preds = tf.nn.softmax(policy_fc_layer)

        #value output layer
        self.value_preds = tf.layers.dense(fc_layer, 1)

        fisher_block = tf.contrib.kfac.fisher_blocks.FullyConnectedKFACBasicFB(layer_collection=self.layer_collection, 
            inputs=fc_layer, outputs=self.value_preds, has_bias=True)
        self.layer_collection.register_block(layer_key="value_fc_layer", fisher_block=fisher_block)

        preds_of_actions_taken = tf.reduce_sum(self.policy_preds * tf.one_hot(self.actions_taken, 
            depth=self.num_actions), axis=1)
        self.loss = -np.log(preds_of_actions_taken) * (self.r_d - self.value_preds) # TODO: incorporate loss?
        self.train_op = tf.contrib.kfac.optimizer.KfacOptimizer(self.args.lr,
            cov_ema_decay=self.args.moving_avg_decay, damping=self.args.damping_lambda,
            layer_collection=self.layer_collection, momentum=self.args.kfac_momentum)
        
        #summaries
        self.lossSummary = tf.summary.scalar("loss", self.loss)
        self.valLossSummary = tf.summary.scalar("val_loss", self.loss)


    # TODO train_step
    def train_step(self, trainImagesLabelsTup, valImagesLabelsTup):
        numBatches = len(trainImagesLabelsTup[0]) / self.args.batch_size

        for epoch in xrange(c.NUM_EPOCHS):
            #train
            print "\n\nEpoch", epoch+1, "out of", c.NUM_EPOCHS, "\n\n"
            for bNum in xrange(numBatches):

                images, labels = trainImagesLabelsTup
                startIndex = bNum*self.args.batch_size
                stopIndex = startIndex + self.args.batch_size
                xBatch = images[startIndex : stopIndex]
                yBatch = labels[startIndex : stopIndex]
                
                #TRAIN STEP
                sessArgs = [self.loss, self.lossSummary, self.preds, self.train_op]
                feedDict = {self.x_Batch: xBatch, 
                            self.y_Batch: yBatch,
                            self.drop_rate}
                loss, summary, preds, _ = self.sess.run(sessArgs, feed_dict=feedDict)

                #write summary (loss)
                # note instead of "tf.train.global_step", really could just ask for it in sessArgs
                self.summary_writer.add_summary(summary, global_step = tf.train.global_step(self.sess, self.glob_step))
                #to do mannualy ...
                #summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss), ])
                
                percentDone = 100*(float(bNum)/numBatches)
                print "\nbatch loss: %.6f \t\t\t %.2f %%" % (loss, percentDone)
                print "first labels:", yBatch[0], yBatch[1]
                print "first preds:", preds[0], preds[1]

                # #validate if enough batches have passed
                # everyXBatches = numBatches / c.VAL_TIMES
                # # +1 to skip validating at start
                # if (bNum+1)%everyXBatches == 0:
                #     print "\nValidating..."
                #     self.eval(valImagesLabelsTup)
            
            #validate at end of epoch
            print "\nValidating..."
            self.validate(valImagesLabelsTup)

        #save model
        self.saver.save(self.sess, self.args.model_save_path, global_step = self.glob_step)


    # TODO
    def validate(self, valImagesLabelsTup):
        #validate
        sessArgs = [self.preds, self.loss, self.valLossSummary]
        feedDict = {self.x_Batch: valImagesLabelsTup[0], 
                    self.y_Batch: valImagesLabelsTup[1], 
                    self.training: False,
                    self.drop_rate: 0.0}
        preds, loss, summary =  self.sess.run(sessArgs, feed_dict=feedDict)
        #write summary
        self.summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.glob_step))
        #print preds and loss
        print preds
        print "\nloss: %.6f \n" % (loss)

    # TODO
    def predict(self):
        pass


