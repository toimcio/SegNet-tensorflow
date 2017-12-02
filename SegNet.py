import json
import os

import tensorflow as tf
import numpy as np
import random
from layers_object import conv_layer, up_sampling, max_pool, initialization, \
    variable_with_weight_decay
from evaluation_object import normal_loss, per_class_acc, get_hist, print_hist_summary, train_op
from inputs_object import get_filename_list, dataset_inputs, get_all_test_data
from drawings_object import draw_plots


class SegNet:
    def __init__(self, conf_file="config.json"):
        with open(conf_file) as f:
            self.config = json.load(f)

        self.num_classes = self.config["NUM_CLASSES"]
        self.use_vgg = self.config["USE_VGG"]

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            self.vgg16_npy_path = self.config["VGG_FILE"]
            self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
            print("VGG parameter loaded")

        self.test_file = self.config["TRAIN_FILE"]
        self.val_file = self.config["VAL_FILE"]
        self.img_prefix = self.config["IMG_PREFIX"]
        self.label_prefix = self.config["LABEL_PREFIX"]
        self.bayes = self.config["BAYES"]
        self.opt = self.config["OPT"]
        self.saved_dir = self.config["SAVE_MODEL_DIR"]
        self.input_w = self.config["INPUT_WIDTH"]
        self.input_h = self.config["INPUT_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.tb_logs = self.config["TB_LOGS"]
        self.batch_size = self.config["BATCH_SIZE"]

        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None
        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session()
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")
            self.inputs_pl = tf.placeholder(tf.float32, [self.batch_size, self.input_h, self.input_w, self.input_c])
            self.labels_pl = tf.placeholder(tf.int64, [self.batch_size, self.input_h, self.input_w, 1])

            # Before enter the images into the architecture, we need to do Local Contrast Normalization
            # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
            # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
            self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
            # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
            self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

            # Second box of convolution layer(4)
            self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

            # Third box of convolution layer(7)
            self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

            # Fourth box of convolution layer(10)
            if self.bayes:
                self.dropout1 = tf.layers.dropout(self.pool3, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout1")
                self.conv4_1 = conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

            # Fifth box of convolution layers(13)
            if self.bayes:
                self.dropout2 = tf.layers.dropout(self.pool4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout2")
                self.conv5_1 = conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

            # ---------------------So Now the encoder process has been Finished--------------------------------------#
            # ------------------Then Let's start Decoder Process-----------------------------------------------------#

            # First box of deconvolution layers(3)
            if self.bayes:
                self.dropout3 = tf.layers.dropout(self.pool5, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout3")
                self.deconv5_1 = up_sampling(self.dropout3, self.pool5_index, self.shape_5, name="unpool_5")
            else:
                self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, name="unpool_5")
            self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)
            # Second box of deconvolution layers(6)
            if self.bayes:
                self.dropout4 = tf.layers.dropout(self.deconv5_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout4")
                self.deconv4_1 = up_sampling(self.dropout4, self.pool4_index, self.shape_4, name="unpool_4")
            else:
                self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, name="unpool_4")
            self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)
            # Third box of deconvolution layers(9)
            if self.bayes:
                self.dropout5 = tf.layers.dropout(self.deconv4_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout5")
                self.deconv3_1 = up_sampling(self.dropout5, self.pool3_index, self.shape_3, name="unpool_3")
            else:
                self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, name="unpool_3")
            self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)
            # Fourth box of deconvolution layers(11)
            if self.bayes:
                self.dropout6 = tf.layers.dropout(self.deconv3_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout6")
                self.deconv2_1 = up_sampling(self.dropout6, self.pool2_index, self.shape_2, name="unpool_2")
            else:
                self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, name="unpool_2")
            self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
            self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)
            # Fifth box of deconvolution layers(13)
            self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, name="unpool_1")
            self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], self.is_training_pl)
            self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], self.is_training_pl)

            with tf.variable_scope('conv_classifier') as scope:
                self.kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64),
                                                         shape=[1, 1, 64, self.num_classes], wd=False)
                self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
                self.biases = variable_with_weight_decay('biases', tf.constant_initializer(0.0),
                                                         shape=[self.num_classes], wd=False)
                self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)

    def retrain(self, max_steps=30001, batch_size=3):
        self.sess = tf.Session()
        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []
        self.train(max_steps=max_steps, batch_size=batch_size)

    def train(self, max_steps=30001, batch_size=3):
        # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET,
        # THE FLAG_OPT SHOULD BE ADAM!!!

        image_filename, label_filename = get_filename_list(self.test_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)

        with self.graph.as_default():
            if self.images_tr is None:
                self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, batch_size, self.config)
                self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size,
                                                                  self.config)

            loss, accuracy, prediction = normal_loss(logits=self.logits, labels=self.labels_pl,
                                                     number_class=self.num_classes)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(tf.global_variables())

            with self.sess.as_default():
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # The queue runners basic reference:
                # https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
                train_writer = tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                for step in range(max_steps):
                    image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: True,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True}

                    _, _loss, _accuracy, summary = self.sess.run([train, loss, accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)
                    print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, self.train_loss[-1],
                                                                                       self.train_accuracy[-1]))

                    if step % 100 == 0:
                        conv_classifier = self.sess.run(self.logits, feed_dict=feed_dict)
                        print('per_class accuracy by logits in training time',
                              per_class_acc(conv_classifier, label_batch, self.num_classes))
                        # per_class_acc is a function from utils
                        train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        print("start validating.......")
                        _val_loss = []
                        _val_acc = []
                        hist = np.zeros((self.num_classes, self.num_classes))
                        for test_step in range(int(20)):
                            fetches_valid = [loss, accuracy, self.logits]
                            image_batch_val, label_batch_val = self.sess.run([self.images_val, self.labels_val])
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               self.is_training_pl: True,
                                               self.keep_prob_pl: 1.0,
                                               self.with_dropout_pl: False}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _val_pred = self.sess.run(fetches_valid, feed_dict_valid)
                            _val_loss.append(_loss)
                            _val_acc.append(_acc)
                            hist += get_hist(_val_pred, label_batch_val)

                        print_hist_summary(hist)

                        self.val_loss.append(np.mean(_val_loss))
                        self.val_acc.append(np.mean(_val_acc))

                        print(
                            "Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                                step, self.train_loss[-1], self.train_accuracy[-1], self.val_loss[-1],
                                self.val_acc[-1]))

                coord.request_stop()
                coord.join(threads)
    
    
    def visual_results(self, dataset_type = "TRAIN", NUM_IMAGES = 3):
        
        #train_dir = "./saved_models/segnet_vgg_bayes/segnet_vgg_bayes_30000/model.ckpt-30000"
        #train_dir = "./saved_models/segnet_scratch/segnet_scratch_30000/model.ckpt-30000"
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:
            
            # Restore saved session
            saver = tf.train.Saver()
            saver.restore(sess, train_dir)
            
            _, _, prediction = normal_loss(logits=self.logits, 
                                           labels=self.labels_pl,
                                           number_class=self.num_classes)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            if (dataset_type=='TRAIN'):
                test_type_path = self.config["TRAIN_FILE"]
                indexes = random.sample(range(367),NUM_IMAGES)
                #indexes = [0,75,150,225,300]
            elif (dataset_type=='VAL'):
                test_type_path = self.config["VAL_FILE"]
                indexes = random.sample(range(101),NUM_IMAGES)
                #indexes = [0,25,50,75,100]
            elif (dataset_type=='TEST'):
                test_type_path = self.config["TEST_FILE"]
                indexes = random.sample(range(233),NUM_IMAGES)
                #indexes = [0,50,100,150,200]

            # Load images
            image_filename,label_filename = get_filename_list(test_type_path, self.config)
            images, labels = get_all_test_data(image_filename,label_filename)

            # Keep images subset of length NUM_IMAGES
            images = [images[i] for i in indexes[0:NUM_IMAGES]]
            labels = [labels[i] for i in indexes[0:NUM_IMAGES]]
            
            num_sample_generate = 30
            pred_tot = []
            var_tot = []
            
            for image_batch, label_batch in zip(images,labels):
                
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
                
                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5}
                    pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                    pred = np.reshape(pred,[image_h,image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: False}
                    prob_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches = [prob], feed_dict = feed_dict) 
                        prob_iter_tot.append(prob_iter_step)

                    prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                    prob_variance = np.var(prob_iter_tot, axis = 0)

                    #THIS TIME I DIDN'T INCLUDE TAU
                    pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label

                    var_sep = [] #var_sep is the corresponding variance if this pixel choose label k
                    length_cur = 0 #length_cur represent how many pixels has been read for one images
                    for row in np.reshape(prob_variance,[image_h*image_w,12]):
                        temp = row[pred[length_cur]]
                        length_cur += 1
                        var_sep.append(temp)
                    var_one = np.reshape(var_sep,[image_h,image_w]) #var_one is the corresponding variance in terms of the "optimal" label
                    pred = np.reshape(pred,[image_h,image_w])

                pred_tot.append(pred)
                var_tot.append(var_one)
            
            draw_plots(images, labels, pred_tot)

    def save(self):
        np.save(self.saved_dir + "Data/trainloss", self.train_loss)
        np.save(self.saved_dir + "Data/trainacc", self.train_accuracy)
        np.save(self.saved_dir + "Data/valloss", self.val_loss)
        np.save(self.saved_dir + "Data/valacc", self.val_acc)
        checkpoint_path = os.path.join(self.saved_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=self.model_version)
        self.model_version += 1
