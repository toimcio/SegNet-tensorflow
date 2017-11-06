import inspect
import os

import numpy as np
import tensorflow as tf
import time
import math
from Inputs import *


def vgg_param_load(vgg16_npy_path): 
    vgg_param_dict = np.load(vgg16_npy_path,encoding='latin1').item()
    for key in vgg_param_dict:
        print(key,vgg_param_dict[key][0].shape,vgg_param_dict[key][1].shape)
    print("vgg parameter loaded")
    return vgg_param_dict
    
vgg_param_dict = vgg_param_load("vgg16.npy")   
               
def inference(images, labels, batch_size, training_state):
    """
    images: is the input images, Training data also Test data
    labels: corresponding labels for images
    batch_size
    phase_train: is utilized to noticify if the parameter should keep as a constant or keep updating 
    """
    
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer(images, "conv1_1",training_state)
    conv1_2 = conv_layer(conv1_1, "conv1_2",training_state)
    pool1,pool1_index,shape_1 = max_pool(conv1_2, 'pool1')
    
    #Second box of covolution layer(4)
    conv2_1 = conv_layer(pool1, "conv2_1",training_state)
    conv2_2 = conv_layer(conv2_1, "conv2_2",training_state)
    pool2,pool2_index,shape_2 = max_pool(conv2_2, 'pool2')
    
    #Third box of covolution layer(7)
    conv3_1 = conv_layer(pool2, "conv3_1",training_state)
    conv3_2 = conv_layer(conv3_1, "conv3_2",training_state)
    conv3_3 = conv_layer(conv3_2, "conv3_3",training_state)
    pool3,pool3_index,shape_3 = max_pool(conv3_3, 'pool3')
    
    #Fourth box of covolution layer(10)
    conv4_1 = conv_layer(pool3, "conv4_1",training_state)
    conv4_2 = conv_layer(conv4_1, "conv4_2",training_state)
    conv4_3 = conv_layer(conv4_2, "conv4_3",training_state)
    pool4,pool4_index,shape_4 = max_pool(conv4_3, 'pool4')

    #Fifth box of covolution layers(13)
    conv5_1 = conv_layer(pool4, "conv5_1",training_state)
    conv5_2 = conv_layer(conv5_1, "conv5_2",training_state)
    conv5_3 = conv_layer(conv5_2, "conv5_3",training_state)
    pool5, pool5_index,shape_5 = max_pool(conv5_3, 'pool5')
        
        
    #---------------------So Now the encoder process has been Finished--------------------------------------#
    #------------------Then Let's start Decoder Process-----------------------------------------------------#
    
    #First box of decovolution layers(3)
    deconv1_1 = up_sampling(pool5, pool5_index,shape=shape_5)
    deconv1_2 = deconv_layer(deconv1_1,[3,3,512,512],shape_5,"deconv1_2",training_state)
    deconv1_3 = deconv_layer(deconv1_2,[3,3,512,512],shape_5,"deconv1_3",training_state)
    deconv1_4 = deconv_layer(deconv1_3,[3,3,512,512],shape_5,"deconv1_4",training_state)
    
    #Second box of deconvolution layers(6)
    deconv2_1 = up_sampling(deconv1_4,pool4_index,shape = shape_4)
    deconv2_2 = deconv_layer(deconv2_1,[3,3,512,512],shape_4,"deconv2_2",training_state)
    deconv2_3 = deconv_layer(deconv2_2,[3,3,512,512],shape_4,"deconv2_3",training_state)
    deconv2_4 = deconv_layer(deconv2_3,[3,3,256,512],[5,45,60,256],"deconv2_4",training_state)
    
    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(deconv2_4,pool3_index,shape = shape_3)
    deconv3_2 = deconv_layer(deconv3_1,[3,3,256,256],shape_3,"deconv3_2",training_state)
    deconv3_3 = deconv_layer(deconv3_2,[3,3,256,256],shape_3,"deconv3_3",training_state)
    deconv3_4 = deconv_layer(deconv3_3,[3,3,128,256],[5,90,120,128],"deconv3_4",training_state)
    
    #Fourth box of deconvolution layers(11)
    deconv4_1 = up_sampling(deconv3_4,pool2_index,shape = shape_2)
    deconv4_2 = deconv_layer(deconv4_1,[3,3,128,128],shape_2,"deconv4_2",training_state)
    deconv4_3 = deconv_layer(deconv4_2,[3,3,64,128],[5,180,240,64],"deconv4_3",training_state)
    
    #Fifth box of deconvolution layers(13)
    deconv5_1 = up_sampling(deconv4_3,pool1_index,shape = shape_1)
    deconv5_2 = deconv_layer(deconv5_1,[3,3,64,64],shape_1,"deconv5_2",training_state)
    deconv5_3 = deconv_layer(deconv5_2,[3,3,11,64],[shape_1[0],shape_1[1],shape_1[2],11],"deconv5_3",training_state)
    
    prob = tf.nn.softmax(deconv5_3,name = "prob")
    
    loss, accuracy, logits, prediction = cal_loss(prob,labels)
    return loss, accuracy, logits, prediction
    


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(inputs,name):
    value,index = tf.nn.max_pool_with_argmax(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name,
                                            Targmax = tf.int32)
    sh_temp = inputs.shape
    shape_inputs = [sh_temp[0].value,sh_temp[1].value,sh_temp[2].value,sh_temp[3].value]                       
    print('value shape',value.shape)
    print('index shape',index.shape)
                           
    
    return value,index,shape_inputs
#here value is the max value, index is the corresponding index, the detail information is here https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/max_pool_with_argmax
    
def conv_layer(bottom, name, training_state):
    with tf.variable_scope(name):
        
        filt = get_conv_filter(name)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)
        out = batch_norm(bias,training_state,name)
            
        relu = tf.nn.relu(out)
        print(relu)
        return relu
        #norm is used to identify if we should use batch normalization!
def batch_norm(bias_input, is_training, scope):
    if is_training is True:
        return tf.contrib.layers.batch_norm(bias_input,is_training = True,center = False,
                                                      scope = scope+"_bn")
    else:
        return tf.contrib.layers.batch_norm(bias_input,is_training = False,center = False,
                                                      scope = scope+"_bn", reuse = True)
#is_training = Ture, it will accumulate the statistics of the movements into moving_mean and moving_variance. When it's 
#not in a training mode, then it would use the values of the moving_mean, and moving_variance. Which is exactly what we want,
#since when it's not training, we are not allowed to use the actual mean and variance for the validation data. 
#reuse is that if we will reuse the layers, which I really don't understand what does that mean

def get_conv_filter(name):
    return tf.constant(vgg_param_dict[name][0], name="filter")
    #so here load the weight for VGG-16, which is kernel, the kernel size for different covolution layers will show in function
    #vgg_param_load

def get_bias(name):
    return tf.constant(vgg_param_dict[name][1], name="biases")
    #here load the bias for VGG-16, the bias size will be 64,128,256,512,512, also shown in function vgg_param_load
    

def unravel_index(indices, shape):
    with tf.name_scope('unravel_index'):
        indices = tf.expand_dims(indices, 0)
        shape = tf.expand_dims(shape, 1)
        strides = tf.cumprod(shape, reverse=True)
        strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
        return (indices % strides) // strides_shifted
    #This function is utilized to transform the flattened maxpooling index to the original 4D tensor, and have already test
    #it, which works brilliant!
    """
    indices: indices will be the output index from maxpooling, just to make sure if it's only 1D!
    Shape: the shape of the original data,[batch_size,height,width,Num_of_Channels]
    output: It's the 4D maxpooling indices!
    """
    
def up_sampling(value,indices,shape):
    """
    Inputs:
    value: the maximum value from maxpooling function, value need to be a tensor. The most important thing for
    value is that it needs to be reshaped to be only one column! 
    indices: the flattened position for the maximum value from maxpooling function. Also indices need to be reshaped
    to be two dimension. [Num_tot_values,4]. 4 is because we have 4 dimension
    shape: the shape of the original data, [batch_size,height,width,Num_of_Channels]
    Outputs:
    up_sample_sp: The sparse matrix from the up_sampling.
    """
    mx_sh = value.shape
    tot = mx_sh[0].value*mx_sh[1].value*mx_sh[2].value*mx_sh[3].value
    value_reshape = tf.reshape(value,[tot])
    index_reshape = tf.reshape(indices,[1,tot])
    print('The shape of reshaped maxvalue',value_reshape.shape)
    print('The shape of reshaped maxindex',index_reshape.shape)
    pooling_index_4d = tf.to_int64(unravel_index(index_reshape,shape),name='ToInt64')
    print('The shape of 4d indices', pooling_index_4d.shape)
    sp_tensor = tf.SparseTensor(tf.reshape(pooling_index_4d,
                                           [pooling_index_4d.shape[2].value,
                                            pooling_index_4d.shape[1].value]),
                                            values = value_reshape, dense_shape = shape)
    sp_dense = tf.sparse_tensor_to_dense(sp_tensor)
    print('The shape of sparse matrix', sp_dense.shape)
    return sp_dense


def _initialization(k,c):
    """
    Here the reference paper is https:arxiv.org/pdf/1502.01852
    k is the filter size
    c is the number of input channels in the filter tensor
    we assume for all the layers, the Kernel Matrix follows a gaussian distribution N(0, \sqrt(2/nl)), where nl is 
    the total number of units in the input, k^2c, k is the spartial filter size and c is the number of input channels. 
    Output:
    The initialized weight
    """
    std = math.sqrt(2. / (k**2 * c))
    return tf.truncated_normal_initializer(stddev = std)



    
def deconv_layer(inputs,kernel_size,output_shape,name,training_state):
    """
    This deconv_layer is utilized to convolve with the upsampled output, and also layer output
    output_shape is different for different layers
    The kernel_size = [height,width,output_channel,input_channel]
    """
    bias_shape = output_shape[-1]
    weight_shape = kernel_size
    k = kernel_size[0]
    c = kernel_size[3]
    with tf.variable_scope(name):
        weights = tf.get_variable(name+"weight",shape=weight_shape,initializer = _initialization(k,c))
        bias = tf.get_variable(name+"bias",shape = bias_shape, initializer = tf.constant_initializer(0.1))
        deconv = tf.nn.conv2d_transpose(inputs,weights,output_shape = output_shape,strides = [1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(deconv, bias)
        out = batch_norm(bias,training_state,name)
         
        relu = tf.nn.relu(out)
        print(relu)
        return relu
    
    
    
def cal_loss(logits,labels):
    loss_weight = np.array([
    0.2595,
    0.1826,
    4.5640,
    0.1417,
    0.9051,
    0.3826,
    9.6446,
    1.8418,
    0.6823,
    6.2478,
    7.3614,
    ]) 
    #class 0 to 11, but the class 11 is ignored, so maybe the class 11 is background!
    
    labels = tf.cast(labels, tf.int32)
    NUM_CLASS = 11
    loss,accuracy,prediction = weighted_loss(logits,labels,number_class = NUM_CLASS, frequency = loss_weight)
    return loss, accuracy, logits, prediction


def weighted_loss(logits,labels,number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the probability of the pixels belongs to class
    labels: true label information 
    number_class: In the CamVid data set, it's 11 classes, or 12, because class 11 seems to be background? 
    frequency: is the frequency of each class
    Outputs:
    Loss
    Accuracy
    """
    label_flatten = tf.reshape(labels,[-1,1])
    label_onehot = tf.reshape(tf.onehot(label_flatten,depth=number_class),[-1,number_class])
    cross_entropy = -tf.reduce_sum(tf.multiply((label_onehot*tf.log(logits+1e-10)),frequency),reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy,name = "cross entropy")
    argmax_logit = tf.to_int32(tf.argmax(logits,axis=1))
    argmax_label = tf.to_int32(tf.argmax(label_onehot,axis=1))
    correct = tf.to_float(tf.equal(argmax_logit,argmax_label))
    accuracy = tf.reduce_mean(correct)
    return loss, accuracy, argmax_logit 

def train_op(total_loss,global_step):
    """
    This part is from the code 'implement slightly different for segnet in Tensorflow', basically the process are same, only
    change some part
    Input:
    total_loss: The loss 
    global_step: Now I have no idea about what is global step, in tensorflow: it says opetional Variable to increment by one 
    after the variable have been updated
    Output
    The train_op
    """
    Learning_Rate = 0.1
    MOVING_AVERAGE_DECAY = 0.999
    optimizer = tf.train.AdamOptimizer(learning_rate = Learning_Rate)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    apply_grad_op = optimizer.apply_gradients(grads_and_vars)
    
    #Add histograms for training variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)
        
    #Add histograms for gradients
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
            
    #Tracking the moving averages of all the trainable variables. Because for some models, using the moving average to 
    #represent the value will improve the performance very well. 
    #The reference is from here https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    
    
    #so here tf.trainable_variables only include the weight, bias for the decoder part, so instead of calling the variables 
    #one by one, it's much easier to call them by trainable_variables, since we have already define the training_phase in our
    #layers.
    with tf.control_dependencies([apply_grad_op]):
        train_op = variable_averages.apply(tf.trainable_varialbes())
        
    return train_op

def test(FLAGS):
    """
    This part is utilized for testing, and the input FLAG including the test_image directory, batch_size,
    training_parameters, and the image height, width, and channels of the image.
    The output should be the prediction labels picture for the test image on the basis of the model parameters
    """
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir
    test_dir = FLAGS.test_dir
    test_ckpt = FLAGS.testing
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    batch_size = 1 #because this is testing, the batch size are supposed to be 1.
    image_filenames,label_filenames=get_filename_list(path)
    test_data_tensor = tf.placeholder(tf.float32,shape=[batch_size,image_w,image_h,image_c])
    test_label_tensor = tf.placeholder(tf.int32,shape=[batch_size,image_w,image_h,1])
    phase_train = tf.placeholder(tf.bool,name = 'phase_train')
    #tf.bool since phase_train is only True or False
    loss, accuracy, prediction = inference(test_data_tensor,test_label_tensor,batch_size,phase_train)
    variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    variables_to_restore = variable_averages.variable_to_restore()
    
    with tf.Session() as sess:
        saver.restore(sess, test_ckpt)
        #reloading the saved sess section from training part
        images,labels = get_all_test_data(image_filenames, label_filenames)
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for image_batch, label_batch in zip(images,labels):
            feed_dict = {test_data_tensor: image_batch,
                        test_label_tensor: label_batch,
                        phase_train: False}
            accuracy,logits,prediction = sess.run([accuracy, logits, prediction], feed_dict = feed_dict)
            
            
            if FLAGS.save_image is True:
                writeImage(prediction[0],'prediction_image')
                
            # hist += get_hist(logits,label_batch)
            #get_hist is a function that is written in the utils file
            
        #accu_total = np.diag(hist).sum()/hist.sum()
        #iu = np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))
        #print("accu:", accu_total)
        #print("mean IU:", np.nanmean(iu))
        #np.nanmean ignore the nan value, calculat the mean along the specific axis
        
        
def TRAINING():
    """
    As before, FLAGS including all the necessary information!
    """
    max_steps = 1
    batch_size = 5
    train_dir = "../SegNet/CamVid/train.txt"
    image_dir = "../SegNet/CamVid/test.txt"
    val_dir = "../SegNet/CamVid/val.txt"
    image_w = 360
    image_h = 480
    image_c = 3
    
    image_filename,label_filename = get_filename_list(image_dir)
    val_image_filename, val_label_filename = get_filename_list(val_dir)
    
    with tf.Graph().as_default():
        
        train_data_tensor = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        train_label_tensor = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        val_data_tensor = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        val_label_tensor = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        
        phase_train = tf.placeholder(tf.bool, name = "phase_train")
        global_step = tf.Variable(0, trainable = False)
        
        images_train, labels_train = CamVidInputs(image_filename, label_filename, batch_size)
        images_val, labels_val = CamVidInputs(val_image_filename, val_label_filename, batch_size)
        
        
        loss, accuracy, logits, prediction = inference(train_data_tensor, train_label_tensor, batch_size, phase_train)
        train_op = train_op(loss, global_step)
        
        saver = tf.train.Saver(tf.global_variables())
        
        summary_op = tf.summary.merge_all()
        #Merges all summaries collected in the default graph.
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
          
        #The queue runners basic reference: https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues/
        #This is utilized to make sure that each image only use once?
            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
            
            train_loss, train_accuracy = [],[]
            val_loss, val_accuracy = [],[]
            
        
            for step in range(max_steps):
                _train_loss, _train_accuracy = [],[]
                image_batch, label_batch = sess.run([images_train,labels_train])
            #But I have to say I run to run this outside of this, but it doesn't read the data, so there must be something 
            #wrong
                feed_dict = {train_data_tensor: image_batch,
                            train_label_tensor: label_batch,
                            phase_train: True}
            
                _, _loss, _accuracy = sess.run([train_op, loss, accuracy], feed_dict = feed_dict)
                
                _train_loss.append(_loss)
                _train_accuracy.append(_accuracy)
                if step % 10 == 0:
                    train_loss.append(np.mean(_train_loss))
                    train_accuracy.append(np.mean(_train_accuracy))
                    
                    pred = sess.run(prediction, feed_dict = feed_dict)
                    per_class_acc(pred, label_batch) 
                    #per_class_acc is a function from utils 
                    
                if step % 100 == 0:
                    print("start validating.......")
                    
                    for test_step in range(int(TEST_ITER)):
                        _val_loss = []
                        _val_acc = []
                        fetches_valid = [loss, accuracy]
                        image_batch_val, label_batch_val = sess.run([images_val, labels_val])
                        feed_dict_valid = {val_data_tensor: image_batch_val,
                                          val_label_tensor: label_batch_val,
                                          phase_train:True}
                        _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                        
                        _val_loss.append(_val_loss)
                        _val_acc.append(_val_acc)
                        
                    val_loss.append(np.mean(_val_loss))
                    val_acc.append(np.mean(_val_acc))
                    
                    print("Epoch {}: Train Loss {:6.3f}, Train Accu{:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                    step, train_loss[-1], train_accuracy[-1], val_loss[-1], val_acc[-1]))
                    
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(train_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step = step)
                    
            #coord.request_stop()
            #coord.join(threads)
            
                    
                
                    
                    
                        
                     
                        
                    

                
                
            
            
        
    
        



        
    






    
    
    




        
    

    
    
    
    



