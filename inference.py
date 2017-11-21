# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:09:39 2017
This file is utilized to denote different inference method. There are four different inference method, segnet_vgg, segnet_scratch, segnet_bayes_vgg, segnet_bayes_scratch
@author: s161488
"""
import numpy as np
import tensorflow as tf
from layers import conv_layer, conv_layer_enc, up_sampling, max_pool, _variable_on_cpu, _initialization,_variable_with_weight_decay

NUM_CLASS = 12

def segnet_vgg(images,labels,batch_size,training_state,keep_prob):
    """
    Train the archietecture by vgg parameters initialization
    images: is the input images, Training data also Test data
    labels: corresponding labels for images
    batch_size
    phase_train: is utilized to noticify if the parameter should keep as a constant or keep updating 
    """
    #Before enter the images into the archetecture, we need to do Local Contrast Normalization 
    #But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
    #Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    norm1 = tf.nn.lrn(images,depth_radius = 5, bias = 1.0, alpha=0.0001,beta=0.75,name = 'norm1')
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer_enc(norm1, "conv1_1", [3,3,3,64], training_state)
    conv1_2 = conv_layer_enc(conv1_1, "conv1_2", [3,3,64,64], training_state)
    pool1,pool1_index,shape_1 = max_pool(conv1_2, 'pool1')
    
    #Second box of covolution layer(4)
    conv2_1 = conv_layer_enc(pool1, "conv2_1", [3,3,64,128], training_state)
    conv2_2 = conv_layer_enc(conv2_1, "conv2_2",[3,3,128,128], training_state)
    pool2,pool2_index,shape_2 = max_pool(conv2_2, 'pool2')
    
    #Third box of covolution layer(7)
    conv3_1 = conv_layer_enc(pool2, "conv3_1", [3,3,128,256], training_state)
    conv3_2 = conv_layer_enc(conv3_1, "conv3_2",[3,3,256,256], training_state)
    conv3_3 = conv_layer_enc(conv3_2, "conv3_3", [3,3,256,256],training_state)
    pool3,pool3_index,shape_3 = max_pool(conv3_3, 'pool3')
    
    #Fourth box of covolution layer(10)
    conv4_1 = conv_layer_enc(pool3, "conv4_1", [3,3,256,512], training_state)
    conv4_2 = conv_layer_enc(conv4_1, "conv4_2", [3,3,512,512], training_state)
    conv4_3 = conv_layer_enc(conv4_2, "conv4_3", [3,3,512,512], training_state)
    pool4,pool4_index,shape_4 = max_pool(conv4_3, 'pool4')

    #Fifth box of covolution layers(13)
    conv5_1 = conv_layer_enc(pool4, "conv5_1", [3,3,512,512], training_state)
    conv5_2 = conv_layer_enc(conv5_1, "conv5_2",[3,3,512,512], training_state)
    conv5_3 = conv_layer_enc(conv5_2, "conv5_3",[3,3,512,512], training_state)
    pool5, pool5_index,shape_5 = max_pool(conv5_3, 'pool5')
        
        
    #---------------------So Now the encoder process has been Finished--------------------------------------#
    #------------------Then Let's start Decoder Process-----------------------------------------------------#
    
    #First box of decovolution layers(3)
    deconv5_1 = up_sampling(pool5, pool5_index,shape_5,name="unpool_5",ksize=[1, 2, 2, 1])
    deconv5_2 = conv_layer(deconv5_1,"deconv5_2",[3,3,512,512], training_state)
    deconv5_3 = conv_layer(deconv5_2,"deconv5_3",[3,3,512,512], training_state)
    deconv5_4 = conv_layer(deconv5_3,"deconv5_4",[3,3,512,512], training_state)
    #Second box of deconvolution layers(6)
    deconv4_1 = up_sampling(deconv5_4,pool4_index, shape_4,name="unpool_4",ksize=[1, 2, 2, 1])
    deconv4_2 = conv_layer(deconv4_1,"deconv4_2", [3,3,512,512], training_state)
    deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3,3,512,512], training_state)
    deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3,3,512,256], training_state)
    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(deconv4_4,pool3_index, shape_3,name="unpool_3",ksize=[1, 2, 2, 1])
    deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3,3,256,256], training_state)
    deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3,3,256,256], training_state)
    deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3,3,256,128], training_state)
    #Fourth box of deconvolution layers(11)
    deconv2_1 = up_sampling(deconv3_4,pool2_index, shape_2,name="unpool_2",ksize=[1, 2, 2, 1])
    deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3,3,128,128], training_state)
    deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3,3,128,64], training_state)
    #Fifth box of deconvolution layers(13)
    deconv1_1 = up_sampling(deconv2_3,pool1_index, shape_1,name="unpool_1",ksize=[1, 2, 2, 1])
    deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3,3,64,64], training_state)
    deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3,3,64,64], training_state)
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[1, 1, 64, NUM_CLASS],initializer=_initialization(1,64),wd=False,enc = False)
        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0),enc = False)
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
   
    return conv_classifier
    

def segnet_scratch(images,labels,batch_size,training_state,keep_prob):
    """
    images: is the input images, Training data also Test data
    labels: corresponding labels for images
    batch_size
    phase_train: is utilized to noticify if the parameter should keep as a constant or keep updating 
    """
    #Before enter the images into the archetecture, we need to do Local Contrast Normalization 
    #But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
    #Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    norm1 = tf.nn.lrn(images,depth_radius = 5, bias = 1.0, alpha=0.0001,beta=0.75,name = 'norm1')
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer(norm1, "conv1_1", [3,3,3,64], training_state)
    conv1_2 = conv_layer(conv1_1, "conv1_2", [3,3,64,64], training_state)
    pool1,pool1_index,shape_1 = max_pool(conv1_2, 'pool1')
    
    #Second box of covolution layer(4)
    conv2_1 = conv_layer(pool1, "conv2_1", [3,3,64,128], training_state)
    conv2_2 = conv_layer(conv2_1, "conv2_2",[3,3,128,128], training_state)
    pool2,pool2_index,shape_2 = max_pool(conv2_2, 'pool2')
    
    #Third box of covolution layer(7)
    conv3_1 = conv_layer(pool2, "conv3_1", [3,3,128,256], training_state)
    conv3_2 = conv_layer(conv3_1, "conv3_2",[3,3,256,256], training_state)
    conv3_3 = conv_layer(conv3_2, "conv3_3", [3,3,256,256],training_state)
    pool3,pool3_index,shape_3 = max_pool(conv3_3, 'pool3')
    
    #Fourth box of covolution layer(10)
    conv4_1 = conv_layer(pool3, "conv4_1", [3,3,256,512], training_state)
    conv4_2 = conv_layer(conv4_1, "conv4_2", [3,3,512,512], training_state)
    conv4_3 = conv_layer(conv4_2, "conv4_3", [3,3,512,512], training_state)
    pool4,pool4_index,shape_4 = max_pool(conv4_3, 'pool4')

    #Fifth box of covolution layers(13)
    conv5_1 = conv_layer(pool4, "conv5_1", [3,3,512,512], training_state)
    conv5_2 = conv_layer(conv5_1, "conv5_2",[3,3,512,512], training_state)
    conv5_3 = conv_layer(conv5_2, "conv5_3",[3,3,512,512], training_state)
    pool5, pool5_index,shape_5 = max_pool(conv5_3, 'pool5')
        
        
    #---------------------So Now the encoder process has been Finished--------------------------------------#
    #------------------Then Let's start Decoder Process-----------------------------------------------------#
    
    #First box of decovolution layers(3)
    deconv5_1 = up_sampling(pool5, pool5_index,shape_5,name="unpool_5",ksize=[1, 2, 2, 1])
    deconv5_2 = conv_layer(deconv5_1,"deconv5_2",[3,3,512,512], training_state)
    deconv5_3 = conv_layer(deconv5_2,"deconv5_3",[3,3,512,512], training_state)
    deconv5_4 = conv_layer(deconv5_3,"deconv5_4",[3,3,512,512], training_state)
    #Second box of deconvolution layers(6)
    deconv4_1 = up_sampling(deconv5_4,pool4_index, shape_4,name="unpool_4",ksize=[1, 2, 2, 1])
    deconv4_2 = conv_layer(deconv4_1,"deconv4_2", [3,3,512,512], training_state)
    deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3,3,512,512], training_state)
    deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3,3,512,256], training_state)
    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(deconv4_4,pool3_index, shape_3,name="unpool_3",ksize=[1, 2, 2, 1])
    deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3,3,256,256], training_state)
    deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3,3,256,256], training_state)
    deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3,3,256,128], training_state)
    #Fourth box of deconvolution layers(11)
    deconv2_1 = up_sampling(deconv3_4,pool2_index, shape_2,name="unpool_2",ksize=[1, 2, 2, 1])
    deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3,3,128,128], training_state)
    deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3,3,128,64], training_state)
    #Fifth box of deconvolution layers(13)
    deconv1_1 = up_sampling(deconv2_3,pool1_index, shape_1,name="unpool_1",ksize=[1, 2, 2, 1])
    deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3,3,64,64], training_state)
    deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3,3,64,64], training_state)
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[1, 1, 64, NUM_CLASS],initializer=_initialization(1,64),wd=False,enc = False)
        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0),enc = False)
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    
    return conv_classifier
    
    
    
def segnet_bayes_scratch(images,labels,batch_size,training_state,keep_prob):
    """
    images: the training and validation image
    labels: corresponding labels
    batch_size:
    training_state:
    keep_prob: for the training time, it's 0.5, but for the validation time is 1.0, all the units are utlized for the validation time. The rate input in tf.layers.dropout
    represent the dropout rate, so the for the validation time, the dropout rate should be 0, which is the reason that keep_prob = 1.
    output:
    logits 
    """
    #Before enter the images into the archetecture, we need to do Local Contrast Normalization 
    #But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
    #Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    norm1 = tf.nn.lrn(images,depth_radius = 5, bias = 1.0, alpha=0.0001,beta=0.75,name = 'norm1')
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer(norm1, "conv1_1", [3,3,3,64], training_state)
    conv1_2 = conv_layer(conv1_1, "conv1_2", [3,3,64,64], training_state)
    pool1,pool1_index,shape_1 = max_pool(conv1_2, 'pool1')
    
    #Second box of covolution layer(4)
    conv2_1 = conv_layer(pool1, "conv2_1", [3,3,64,128], training_state)
    conv2_2 = conv_layer(conv2_1, "conv2_2",[3,3,128,128], training_state)
    pool2,pool2_index,shape_2 = max_pool(conv2_2, 'pool2')
    
    #Third box of covolution layer(7)
    conv3_1 = conv_layer(pool2, "conv3_1", [3,3,128,256], training_state)
    conv3_2 = conv_layer(conv3_1, "conv3_2",[3,3,256,256], training_state)
    conv3_3 = conv_layer(conv3_2, "conv3_3", [3,3,256,256],training_state)
    pool3,pool3_index,shape_3 = max_pool(conv3_3, 'pool3')
    dropout1 = tf.layers.dropout(pool3, rate=(1-keep_prob), training = training_state, name = "dropout1")

    #Fourth box of covolution layer(10)
    conv4_1 = conv_layer(dropout1, "conv4_1", [3,3,256,512], training_state)
    conv4_2 = conv_layer(conv4_1, "conv4_2", [3,3,512,512], training_state)
    conv4_3 = conv_layer(conv4_2, "conv4_3", [3,3,512,512], training_state)
    pool4,pool4_index,shape_4 = max_pool(conv4_3, 'pool4')
    dropout2 = tf.layers.dropout(pool4,rate = (1-keep_prob), training = training_state,name = "dropout2")

    #Fifth box of covolution layers(13)
    conv5_1 = conv_layer(dropout2, "conv5_1", [3,3,512,512], training_state)
    conv5_2 = conv_layer(conv5_1, "conv5_2",[3,3,512,512], training_state)
    conv5_3 = conv_layer(conv5_2, "conv5_3",[3,3,512,512], training_state)
    pool5, pool5_index,shape_5 = max_pool(conv5_3, 'pool5')
    dropout3 = tf.layers.dropout(pool5,rate = (1-keep_prob), training = training_state,name =  "dropout3")
  
        
    #---------------------So Now the encoder process has been Finished--------------------------------------#
    #------------------Then Let's start Decoder Process-----------------------------------------------------#
    
    #First box of decovolution layers(3)
    deconv5_1 = up_sampling(dropout3, pool5_index,shape_5,name="unpool_5",ksize=[1, 2, 2, 1])
    deconv5_2 = conv_layer(deconv5_1,"deconv5_2",[3,3,512,512], training_state)
    deconv5_3 = conv_layer(deconv5_2,"deconv5_3",[3,3,512,512], training_state)
    deconv5_4 = conv_layer(deconv5_3,"deconv5_4",[3,3,512,512], training_state)
    dropout4 = tf.layers.dropout(deconv5_4, rate = (1-keep_prob),training = training_state, name = "dropout4")

    #Second box of deconvolution layers(6)
    deconv4_1 = up_sampling(dropout4,pool4_index, shape_4,name="unpool_4",ksize=[1, 2, 2, 1])
    deconv4_2 = conv_layer(deconv4_1,"deconv4_2", [3,3,512,512], training_state)
    deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3,3,512,512], training_state)
    deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3,3,512,256], training_state)
    dropout5 = tf.layers.dropout(deconv4_4,rate = (1-keep_prob), training = training_state,name =  "dropout5")

    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(dropout5,pool3_index, shape_3,name="unpool_3",ksize=[1, 2, 2, 1])
    deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3,3,256,256], training_state)
    deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3,3,256,256], training_state)
    deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3,3,256,128], training_state)
    dropout6 = tf.layers.dropout(deconv3_4,rate =(1-keep_prob),training = training_state,name =  "dropout6")

    #Fourth box of deconvolution layers(11)
    deconv2_1 = up_sampling(dropout6,pool2_index, shape_2,name="unpool_2",ksize=[1, 2, 2, 1])
    deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3,3,128,128], training_state)
    deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3,3,128,64], training_state)
    #Fifth box of deconvolution layers(13)
    deconv1_1 = up_sampling(deconv2_3,pool1_index, shape_1,name="unpool_1",ksize=[1, 2, 2, 1])
    deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3,3,64,64], training_state)
    deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3,3,64,64], training_state)
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[1, 1, 64, NUM_CLASS],initializer=_initialization(1,64),wd=False,enc = False)
        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0),enc = False)
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    
    return conv_classifier
    
    
def segnet_bayes_vgg(images,labels,batch_size,training_state,keep_prob):
    """
    images: the training and validation image
    labels: corresponding labels
    batch_size:
    training_state:
    keep_prob: for the training time, it's 0.5, but for the validation time is 1.0, all the units are utlized for the validation time. The rate input in tf.layers.dropout
    represent the dropout rate, so the for the validation time, the dropout rate should be 0, which is the reason that keep_prob = 1.
    output:
    logits 
    """
    #Before enter the images into the archetecture, we need to do Local Contrast Normalization 
    #But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
    #Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
    norm1 = tf.nn.lrn(images,depth_radius = 5, bias = 1.0, alpha=0.0001,beta=0.75,name = 'norm1')
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer_enc(norm1, "conv1_1", [3,3,3,64], training_state)
    conv1_2 = conv_layer_enc(conv1_1, "conv1_2", [3,3,64,64], training_state)
    pool1,pool1_index,shape_1 = max_pool(conv1_2, 'pool1')
    
    #Second box of covolution layer(4)
    conv2_1 = conv_layer_enc(pool1, "conv2_1", [3,3,64,128], training_state)
    conv2_2 = conv_layer_enc(conv2_1, "conv2_2",[3,3,128,128], training_state)
    pool2,pool2_index,shape_2 = max_pool(conv2_2, 'pool2')
    
    #Third box of covolution layer(7)
    conv3_1 = conv_layer_enc(pool2, "conv3_1", [3,3,128,256], training_state)
    conv3_2 = conv_layer_enc(conv3_1, "conv3_2",[3,3,256,256], training_state)
    conv3_3 = conv_layer_enc(conv3_2, "conv3_3", [3,3,256,256],training_state)
    pool3,pool3_index,shape_3 = max_pool(conv3_3, 'pool3')
    dropout1 = tf.layers.dropout(pool3, rate=(1-keep_prob), training = training_state, name = "dropout1")

    #Fourth box of covolution layer(10)
    conv4_1 = conv_layer_enc(dropout1, "conv4_1", [3,3,256,512], training_state)
    conv4_2 = conv_layer_enc(conv4_1, "conv4_2", [3,3,512,512], training_state)
    conv4_3 = conv_layer_enc(conv4_2, "conv4_3", [3,3,512,512], training_state)
    pool4,pool4_index,shape_4 = max_pool(conv4_3, 'pool4')
    dropout2 = tf.layers.dropout(pool4,rate = (1-keep_prob), training = training_state,name = "dropout2")

    #Fifth box of covolution layers(13)
    conv5_1 = conv_layer_enc(dropout2, "conv5_1", [3,3,512,512], training_state)
    conv5_2 = conv_layer_enc(conv5_1, "conv5_2",[3,3,512,512], training_state)
    conv5_3 = conv_layer_enc(conv5_2, "conv5_3",[3,3,512,512], training_state)
    pool5, pool5_index,shape_5 = max_pool(conv5_3, 'pool5')
    dropout3 = tf.layers.dropout(pool5,rate = (1-keep_prob), training = training_state,name =  "dropout3")
  
        
    #---------------------So Now the encoder process has been Finished--------------------------------------#
    #------------------Then Let's start Decoder Process-----------------------------------------------------#
    
    #First box of decovolution layers(3)
    deconv5_1 = up_sampling(dropout3, pool5_index,shape_5,name="unpool_5",ksize=[1, 2, 2, 1])
    deconv5_2 = conv_layer(deconv5_1,"deconv5_2",[3,3,512,512], training_state)
    deconv5_3 = conv_layer(deconv5_2,"deconv5_3",[3,3,512,512], training_state)
    deconv5_4 = conv_layer(deconv5_3,"deconv5_4",[3,3,512,512], training_state)
    dropout4 = tf.layers.dropout(deconv5_4, rate = (1-keep_prob),training = training_state, name = "dropout4")

    #Second box of deconvolution layers(6)
    deconv4_1 = up_sampling(dropout4,pool4_index, shape_4,name="unpool_4",ksize=[1, 2, 2, 1])
    deconv4_2 = conv_layer(deconv4_1,"deconv4_2", [3,3,512,512], training_state)
    deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3,3,512,512], training_state)
    deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3,3,512,256], training_state)
    dropout5 = tf.layers.dropout(deconv4_4,rate = (1-keep_prob), training = training_state,name =  "dropout5")

    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(dropout5,pool3_index, shape_3,name="unpool_3",ksize=[1, 2, 2, 1])
    deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3,3,256,256], training_state)
    deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3,3,256,256], training_state)
    deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3,3,256,128], training_state)
    dropout6 = tf.layers.dropout(deconv3_4,rate =(1-keep_prob),training = training_state,name =  "dropout6")

    #Fourth box of deconvolution layers(11)
    deconv2_1 = up_sampling(dropout6,pool2_index, shape_2,name="unpool_2",ksize=[1, 2, 2, 1])
    deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3,3,128,128], training_state)
    deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3,3,128,64], training_state)
    #Fifth box of deconvolution layers(13)
    deconv1_1 = up_sampling(deconv2_3,pool1_index, shape_1,name="unpool_1",ksize=[1, 2, 2, 1])
    deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3,3,64,64], training_state)
    deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3,3,64,64], training_state)
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[1, 1, 64, NUM_CLASS],initializer=_initialization(1,64),wd=False,enc = False)
        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0),enc = False)
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    
    return conv_classifier