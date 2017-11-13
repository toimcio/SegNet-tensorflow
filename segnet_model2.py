import inspect
import os

import numpy as np
import tensorflow as tf
import time
import math
from Inputs import *



vgg16_npy_path = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/vgg16.npy"
def vgg_param_load(vgg16_npy_path): 
    vgg_param_dict = np.load(vgg16_npy_path,encoding='latin1').item()
    for key in vgg_param_dict:
        print(key,vgg_param_dict[key][0].shape,vgg_param_dict[key][1].shape)
    print("vgg parameter loaded")
    return vgg_param_dict
    
vgg_param_dict = vgg_param_load(vgg16_npy_path)   

NUM_CLASS = 12
               
def inference(images, labels, batch_size, training_state):
    """
    images: is the input images, Training data also Test data
    labels: corresponding labels for images
    batch_size
    phase_train: is utilized to noticify if the parameter should keep as a constant or keep updating 
    """
    
    #first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
    conv1_1 = conv_layer(images, "conv1_1", [3,3,3,64], training_state)
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
    deconv1_1 = up_sampling(pool5, pool5_index,shape=shape_5)
    #deconv1_2 = deconv_layer(deconv1_1,[3,3,512,512],shape_5,"deconv1_2",training_state)
    #deconv1_3 = deconv_layer(deconv1_2,[3,3,512,512],shape_5,"deconv1_3",training_state)
    #deconv1_4 = deconv_layer(deconv1_3,[3,3,512,512],shape_5,"deconv1_4",training_state)
    deconv1_2 = conv_layer(deconv1_1,"deconv1_2",[3,3,512,512], training_state)
    deconv1_3 = conv_layer(deconv1_2,"deconv1_3",[3,3,512,512], training_state)
    deconv1_4 = conv_layer(deconv1_3,"deconv1_4",[3,3,512,512], training_state)
    #Second box of deconvolution layers(6)
    deconv2_1 = up_sampling(deconv1_4,pool4_index,shape = shape_4)
    #deconv2_2 = deconv_layer(deconv2_1,[3,3,512,512],shape_4,"deconv2_2",training_state)
    #deconv2_3 = deconv_layer(deconv2_2,[3,3,512,512],shape_4,"deconv2_3",training_state)
    #deconv2_4 = deconv_layer(deconv2_3,[3,3,256,512],[shape_4[0],shape_4[1],shape_4[2],256],"deconv2_4",training_state)
    deconv2_2 = conv_layer(deconv2_1,"deconv2_2", [3,3,512,512], training_state)
    deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3,3,512,512], training_state)
    deconv2_4 = conv_layer(deconv2_3, "deconv2_4", [3,3,512,256], training_state)
    #Third box of deconvolution layers(9)
    deconv3_1 = up_sampling(deconv2_4,pool3_index,shape = shape_3)
    #deconv3_2 = deconv_layer(deconv3_1,[3,3,256,256],shape_3,"deconv3_2",training_state)
    #deconv3_3 = deconv_layer(deconv3_2,[3,3,256,256],shape_3,"deconv3_3",training_state)
    #deconv3_4 = deconv_layer(deconv3_3,[3,3,128,256],[shape_3[0],shape_3[1],shape_3[2],128],"deconv3_4",training_state)
    deconv3_2 = conv_layer(deconv3_1,"deconv3_2", [3,3,256,256], training_state)
    deconv3_3 = conv_layer(deconv3_2,"deconv3_3", [3,3,256,256], training_state)
    deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3,3,256,128], training_state)
    #Fourth box of deconvolution layers(11)
    deconv4_1 = up_sampling(deconv3_4,pool2_index,shape = shape_2)
    #deconv4_2 = deconv_layer(deconv4_1,[3,3,128,128],shape_2,"deconv4_2",training_state)
    #deconv4_3 = deconv_layer(deconv4_2,[3,3,64,128],[shape_2[0],shape_2[1],shape_2[2],64],"deconv4_3",training_state)
    deconv4_2 = conv_layer(deconv4_1, "deconv4_2", [3,3,128,128], training_state)
    deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3,3,128,64], training_state)
    #Fifth box of deconvolution layers(13)
    deconv5_1 = up_sampling(deconv4_3,pool1_index,shape = shape_1)
    #deconv5_2 = deconv_layer(deconv5_1,[3,3,64,64],shape_1,"deconv5_2",training_state)
    #deconv5_3 = deconv_layer(deconv5_2,[3,3,11,64],[shape_1[0],shape_1[1],shape_1[2],11],"deconv5_3",training_state)
    deconv5_2 = conv_layer(deconv5_1, "deconv5_2", [3,3,64,64], training_state)
    deconv5_3 = conv_layer(deconv5_2, "deconv5_3", [3,3,64,64], training_state)
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, NUM_CLASS],
                                           initializer=_initialization(1,64),
                                           wd=0.0005,enc = False)
        conv = tf.nn.conv2d(deconv5_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [NUM_CLASS], tf.constant_initializer(0.0),enc = False)
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    print(conv_classifier)
    prob = tf.nn.softmax(conv_classifier,name = "prob")
    print("prob", prob)
    
    loss, accuracy, logits, prediction = cal_loss(prob,labels)
    return loss, accuracy, logits, prediction
    


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(inputs,name):
    value,index = tf.nn.max_pool_with_argmax(tf.to_double(inputs),ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    print('value shape',value.shape)
    print('index shape',index.shape)
    
    return tf.to_float(value),index,inputs.get_shape().as_list()
#here value is the max value, index is the corresponding index, the detail information is here https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/max_pool_with_argmax
    
def conv_layer(bottom, name, shape, training_state):
    """
    Inputs:
    bottom: The input image or tensor
    name: corresponding layer's name
    shape: the shape of kernel size
    training_state: represent if the weight should update 
    Output:
    The output from layers
    """
    kernel_size = shape[0]
    input_channel = shape[2]
    out_channel = shape[3]
    with tf.variable_scope(name):
        #filt = get_conv_filter(name)
        filt = _variable_with_weight_decay('weights',shape = shape, initializer = _initialization(kernel_size,input_channel), wd = False,enc = False)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        #conv_biases = get_bias(name)
        conv_biases = tf.get_variable(name+"bias",shape = out_channel, initializer = tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, conv_biases)
        out = batch_norm(bias,training_state,name)
            
        relu = tf.nn.relu(out)
        print(relu)
        return relu
        #norm is used to identify if we should use batch normalization!
        
def conv_layer_enc(bottom, name, shape, training_state):
    """
    Inputs:
    bottom: The input image or tensor
    name: corresponding layer's name
    shape: the shape of kernel size
    training_state: represent if the weight should update 
    Output:
    The output from layers
    """
    #kernel_size = shape[0]
    #input_channel = shape[2]
    #out_channel = shape[3]
    with tf.variable_scope(name):
        init = get_conv_filter(name)
        #init = tf.constant(init)
        filt = _variable_with_weight_decay('weights',shape = shape, initializer = init, wd = False,enc = True)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases_init = get_bias(name)
        #conv_biases_init = tf.contant(conv_biases_init)
        conv_biases = tf.get_variable(name+"bias", initializer = conv_biases_init)
        bias = tf.nn.bias_add(conv, conv_biases)
        out = batch_norm(bias,training_state,name)
            
        relu = tf.nn.relu(out)
        print(relu)
        return relu
def batch_norm(bias_input, is_training, scope):
    if is_training is True:
        return tf.contrib.layers.batch_norm(bias_input,is_training = True,center = False,
                                                      scope = scope+"_bn")
    else:
        return tf.contrib.layers.batch_norm(bias_input,is_training = False,center = False,
                                                      scope = scope+"_bn", reuse = False)
#is_training = Ture, it will accumulate the statistics of the movements into moving_mean and moving_variance. When it's 
#not in a training mode, then it would use the values of the moving_mean, and moving_variance. 
#shadow_variable = decay * shadow_variable + (1 - decay) * variable, shadow_variable, I think it's the accumulated moving
#average, and then variable is the average for this specific batch of data. For the training part, we need to set is_training
#to be True, but for the validation part, actually we should set it to be False!

def get_conv_filter(name):
    return tf.constant(vgg_param_dict[name][0], name="filter")
    #so here load the weight for VGG-16, which is kernel, the kernel size for different covolution layers will show in function
    #vgg_param_load

def get_bias(name):
    return tf.constant(vgg_param_dict[name][1], name="biases")
    #here load the bias for VGG-16, the bias size will be 64,128,256,512,512, also shown in function vgg_param_load
    

def unravel_index(indices, shape):
    with tf.name_scope('unravel_index'):
        indices = tf.to_int64(tf.expand_dims(indices, 0))
        shape = tf.to_int64(tf.expand_dims(shape, 1))
        strides = tf.to_int64(tf.cumprod(shape, reverse=True))
        strides_shifted = tf.to_int64(tf.cumprod(shape, exclusive=True, reverse=True))
        return (indices % strides) // strides_shifted
    #This function is utilized to transform the flattened maxpooling index to the original 4D tensor, and have already test
    #it, which works brilliant!
    """
    indices: indices will be the output index from maxpooling, just to make sure if it's only 1D!
    Shape: the shape of the original data,[batch_size,height,width,Num_of_Channels]
    output: It's the 4D maxpooling indices!
    """
    
def up_sampling(max_values,max_indices,shape):
    """
    Inputs:
    max_value: the maximum value from maxpooling function, value need to be a tensor. The most important thing for
    value is that it needs to be reshaped to be only one column! 
    max_indices: the flattened position for the maximum value from maxpooling function. Also indices need to be reshaped
    to be two dimension. [Num_tot_values,4]. 4 is because we have 4 dimension
    shape: the shape of the original data, [batch_size,height,width,Num_of_Channels]
    Outputs:
    sp_dense: The sparse matrix from the up_sampling.
    """
    values_reshape = tf.reshape(max_values,[-1])
    indices_reshape = tf.reshape(max_indices,[-1])
    print('The shape of reshaped maxindex',indices_reshape.shape)
    pooling_index_4d = tf.stack(tf.unstack(unravel_index(indices_reshape,shape), axis=0), axis=1)
    print('The shape of 4d indices', pooling_index_4d.shape)
    sp_tensor = tf.SparseTensor(pooling_index_4d, values = values_reshape, dense_shape = shape)
    sp_dense = tf.sparse_tensor_to_dense(sp_tensor, validate_indices=False)
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



    
#def deconv_layer(inputs,kernel_size,output_shape,name,training_state):
#    """
#    This deconv_layer is utilized to convolve with the upsampled output, and also layer output
#    output_shape is different for different layers
#    The kernel_size = [height,width,output_channel,input_channel]
#    """
#    bias_shape = output_shape[-1]
#    weight_shape = kernel_size
#    k = kernel_size[0]
#    c = kernel_size[3]
#    with tf.variable_scope(name):
#        weights = tf.get_variable(name+"weight",shape=weight_shape,initializer = _initialization(k,c))
#        bias = tf.get_variable(name+"bias",shape = bias_shape, initializer = tf.constant_initializer(0.1))
#        deconv = tf.nn.conv2d_transpose(inputs,weights,output_shape = output_shape,strides = [1,1,1,1],padding='SAME')
#        bias = tf.nn.bias_add(deconv, bias)
#        out = batch_norm(bias,training_state,name)
#         
#        relu = tf.nn.relu(out)
#        print(relu)
#        return relu
    
def _variable_on_cpu(name,shape,initializer,enc):
    """
    Help to create a variable which can be stored on CPU memory
    Inputs: 
    name: corresponding layers name, conv1 or ...
    shape: the shape of the weight or bias
    initializer: The initializer from _ínitialization
    Outputs:
    Variable tensor
    """
    with tf.device('/cpu:0'):
        if enc is True:
            var = tf.get_variable(name,initializer = initializer)
        else:
            var = tf.get_variable(name,shape,initializer = initializer)
    return var 

def _variable_with_weight_decay(name,shape,initializer,wd,enc):
    """
    Help to create an initialized variable with weight decay
    The variable is initialized with a truncated normal distribution, only the value for standard deviation is determined
    by the specific function _initialization
    Inputs: wd is utilized to noticify if weight decay is utilized 
    Return: varialbe tensor
    """
    var = _variable_on_cpu(name,shape,initializer,enc)
    if wd is True:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    #tf.nn.l2_loss is utilized to compute half L2 norm of a tensor without the sqrt output = sum(t**2)/2    
    return var
    

    
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
    1.0974
    ]) 
    #class 0 to 11, but the class 11 is ignored, so maybe the class 11 is background!
    
    labels = tf.cast(labels, tf.int32)
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
    label_onehot = tf.reshape(tf.one_hot(label_flatten,depth=number_class),[-1,number_class])
    cross_entropy = -tf.reduce_sum(tf.multiply((label_onehot*tf.log(tf.reshape(logits,[-1,number_class])+1e-10)),frequency),reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy,name = "cross_entropy")
    argmax_logit = tf.to_int32(tf.argmax(logits,axis= -1))
    argmax_label = tf.to_int32(tf.argmax(label_onehot,axis= -1))
    correct = tf.to_float(tf.equal(tf.reshape(argmax_logit,[-1]),argmax_label))
    accuracy = tf.reduce_mean(correct)
    return loss, accuracy, argmax_logit 
    
def Normal_Loss(logits,labels,number_class):
    """
    Calculate the normal loss instead of median frequency balancing
    Inputs: logits, value should be in the interval of [0,1]
    lables: the atual label information'
    number_class:12
    Output:loss,and accuracy
    """
    label_flatten = tf.reshape(labels,[-1,1])
    label_onehot = tf.reshape(tf.one_hot(label_flatten,depth = number_class),[-1,number_class])
    cross_entropy = -tf.reduce_sum(tf.multiply(label_onehot*tf.log(tf.reshape(logits,[-1,number_class])+1e-10)),reduction_indices = [1])
    loss = tf.reduce_mean(cross_entropy)
    
    return loss
    

    

def train_op(total_loss,global_step):
    """
    This part is from the code 'implement slightly different for segnet in Tensorflow', basically the process are same, only
    change some part
    Input:
    total_loss: The loss 
    global_step: global_step is used to track how many batches had been passed. In the training process, the intial
    value for global_step is 0 (tf.variable(0,trainable=False)), then after one batch of images passed, the loss is
    passed into the optimizor to update the weight, then the global step increased by one. Number of batches seen 
    by the graph.. Reference: https://stackoverflow.com/questions/41166681/what-does-tensorflow-global-step-mean
    Output
    The train_op
    """
    Learning_Rate = 0.01
   # MOVING_AVERAGE_DECAY = 0.99
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = Learning_Rate)
    optimizer = tf.train.AdamOptimizer(learning_rate = Learning_Rate)
    #grads_and_vars = optimizer.compute_gradients(total_loss)
    #apply_grad_op = optimizer.apply_gradients(grads_and_vars,global_step = global_step)
    
    #Add histograms for training variables
    #for var in tf.trainable_variables():
    #    tf.summary.histogram(var.op.name,var)
        
    #Add histograms for gradients
    #for grad, var in grads_and_vars:
    #    if grad is not None:
    #        tf.summary.histogram(var.op.name + '/gradients', grad)
            
    #Tracking the moving averages of all the trainable variables. Because for some models, using the moving average to 
    #represent the value will improve the performance very well. 
    #The reference is from here https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
    #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    #Create shadow variables, and add ops to maintain moving averages for the trainable_variables
    #shadow_variable = decay*shawdow_variable+(1-decay)*variables
    #maintain_averages_op = variable_averages.apply(tf.trainable_variables())
    #print("trainable_variable",tf.trainable_variables())
    #To make sure the trainable variables only include the weight and bias for the decoder part
    
    #so here tf.trainable_variables only include the weight, bias for the decoder part, so instead of calling the variables 
    #one by one, it's much easier to call them by trainable_variables, since we have already define the training_phase in our
    #layers.
    #with tf.control_dependencies([apply_grad_op]):
        #training_op = tf.group(maintain_averages_op)
    training_op = optimizer.minimize(total_loss,var_list = tf.trainable_variables(), global_step = global_step)    
    return training_op

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
    max_steps = 10000
    batch_size = 5
    train_dir = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/"
    image_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/train.txt"
    val_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/val.txt"
    image_w = 360
    image_h = 480
    image_c = 3
    
    
    image_filename,label_filename = get_filename_list(image_dir)
    val_image_filename, val_label_filename = get_filename_list(val_dir)
    

    #tf.reset_default_graph()
    with tf.Graph().as_default():
    #with tf.device('/device:GPU:0'):
        
        train_data_tensor = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        train_label_tensor = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        
        
        phase_train = tf.placeholder(tf.bool, name = "phase_train")
        global_step = tf.Variable(0, trainable = False)
        
        min_queue_train = 300
        min_queue_val = 90
        images_train, labels_train = CamVidInputs(image_filename, label_filename, batch_size, min_queue_train)
        images_val, labels_val = CamVidInputs(val_image_filename, val_label_filename, batch_size,min_queue_val)
        
        loss, accuracy, logits, prediction = inference(train_data_tensor, train_label_tensor, batch_size, phase_train)
        train = train_op(loss, global_step)
        
        saver = tf.train.Saver(tf.global_variables())
        
        #summary_op = tf.summary.merge_all()
        #Merges all summaries collected in the default graph.
        
        #config = tf.ConfigProto(log_device_placement = True)
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.8
        #config.operation_timeout_in_ms = 10000
        with tf.Session() as sess:
        #with tf.device('/device:GPU:2'):
            init = tf.global_variables_initializer()
            sess.run(init)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)            
            
        #The queue runners basic reference: https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
            train_loss, train_accuracy = [],[]
            val_loss, val_acc = [],[]
            #_train_loss, _train_accuracy = [],[]
        
            for step in range(max_steps):
                

                image_batch, label_batch = sess.run([images_train,labels_train])

                feed_dict = {train_data_tensor: image_batch,
                            train_label_tensor: label_batch,
                            phase_train: True}
            
                _, _loss, _accuracy = sess.run([train, loss, accuracy], feed_dict = feed_dict)
                
                train_loss.append(_loss)
                train_accuracy.append(_accuracy)
                #print('trainable_variables',tf.trainable_variables())
                #print('probility',logits)
                #if step % 10 == 0:
                    #print("Training is On......")
                    #train_loss.append(np.mean(_train_loss))
                    #train_accuracy.append(np.mean(_train_accuracy))
                    
                    #pred = sess.run(prediction, feed_dict = feed_dict)
                    #print("Iteration {}: Train Loss {:6.3f}, Train Accu{:6.3f}".format(step, train_loss[-1], train_accuracy[-1]))
                    #per_class_acc(pred, label_batch) 
                    #per_class_acc is a function from utils 
                   
                if step % 100 == 0:
                    print("start validating.......")
                    _val_loss = []
                    _val_acc = []
                    for test_step in range(int(20)):
                        
                        fetches_valid = [loss, accuracy]
                        image_batch_val, label_batch_val = sess.run([images_val, labels_val])
                        feed_dict_valid = {train_data_tensor: image_batch_val,
                                          train_label_tensor: label_batch_val,
                                          phase_train:True}
                                          #since we still using mini-batch, so in the batch norm we set phase_train to be
                                          #true, and because we didin't run the trainop process, so it will not update
                                          #the weight!
                        _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                        
                        _val_loss.append(_loss)
                        _val_acc.append(_acc)
                        
                    #train_loss.append(np.mean(_train_loss))
                    #train_accuracy.append(np.mean(_train_accuracy))
                    val_loss.append(np.mean(_val_loss))
                    val_acc.append(np.mean(_val_acc))
                    
                    print("Iteration {}: Train Loss {:6.3f}, Train Accu{:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                    step, train_loss[-1], train_accuracy[-1], val_loss[-1], val_acc[-1]))
                    
                if step == (max_steps-1):
                    return train_loss, train_accuracy, val_loss, val_acc
                    checkpoint_path = os.path.join(train_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step = step) 


                    
            coord.request_stop()
            coord.join(threads)
            
            

             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
        
        

            
              
