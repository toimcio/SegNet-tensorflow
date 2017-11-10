# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:31:14 2017
This script is used to test the way to calculate weighted_loss is correct. 
@author: s161488
"""
import tensorflow as tf
import numpy as np

def weighted_loss(logits,labels,number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the probability of the pixels belongs to class, The shape should
    be [NumBatch, Height, Width,Num_Class]
    labels: the shape of labels is [NumBatch,Height,Width,1] 
    number_class: In the CamVid data set, it's 11 classes, or 12, because class 11 seems to be background? 
    frequency: is the frequency of each class
    Outputs:
    Loss
    Accuracy
    """
    label_flatten = tf.reshape(labels,[-1,1])
    label_onehot = tf.reshape(tf.one_hot(label_flatten,depth=number_class),[-1,number_class])
    #tf.one_hot will create a matrix which is Number_of_Pixels-by-Number_of_class, since the label_flatten
    #shape is Number_of_pixels-by-1, so it will make the tf.one_hot to be 3 dimension, that's the reason
    #we reshape it to be [-1,number_class], otherwise it will be a three dimension.
    print(label_flatten)
    print(label_onehot)
    cross_entropy = -tf.reduce_sum(tf.multiply((label_onehot*tf.log(tf.reshape(logits,[-1,number_class])+1e-10)),
                                               frequency),reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy,name = "cross_entropy")
    argmax_logit = tf.to_int32(tf.argmax(logits,axis= -1))
    argmax_label = tf.to_int32(tf.argmax(label_onehot,axis= -1))
    correct = tf.to_float(tf.equal(tf.reshape(argmax_logit,[-1]),tf.reshape(argmax_label,[-1])))
    accuracy = tf.reduce_mean(correct)
    return loss, accuracy, argmax_logit
    


def Test():
    #logits = np.random.rand(2,2,1,3)
    #logits = np.float32(logits)

    #labels = np.random.randint(low = 0, high = 3, size= [2,2,1,1])
    logits = np.array([[[[0.891667,0.061923,0.190263]],
                    [[0.75866,0.020426,0.7821445]]],
                  [[[0.256981,0.32574216,0.311162]],
                   [[0.82633,0.60955,0.07444312]]]])
    logits = np.float32(logits)
    print(logits)
    labels = np.array([[[[1]],[[1]]],[[[0]],[[0]]]])
    
    loss_weight = np.array([
    0.2595,
    0.1826,
    4.5640,
    ]) 
    number_class = 3
    with tf.Session() as sess:
        loss,acc,label=sess.run(weighted_loss(logits,labels,number_class,loss_weight))
#        fetches = [loss, accuracy, argmax_logit]
#        feed_dict = {logits : logits,labels : labels,number_class : number_class, frequency : loss_weight}
#        lost, acc, label = sess.run()

    return logits,labels,loss,acc,label
    
#The generated data are 
label= np.array([[[0],[2]],[[1],[0]]])
loss = 0.40513584
acc = 0.25
cross_entro = (np.log(0.061923)*0.1826+
              np.log(0.020426)*0.1826+
              np.log(0.256981)*0.2595+
              np.log(0.82633)*0.2595)/4
cross_entro = -((-2.78186-3.890946)*0.1826+(-1.35875-0.19076)*0.2595)
                   
    
    

    

