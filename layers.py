"""
Created on Tue Nov 21 10:09:39 2017
This file is utilized to denote different layers, there are conv_layer, conv_layer_enc, max_pool, up_sampling
@author: s161488
"""
import numpy as np
import tensorflow as tf
import math

vgg16_npy_path = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/vgg16.npy"


def vgg_param_load(vgg16_npy_path): 
    vgg_param_dict = np.load(vgg16_npy_path,encoding='latin1').item()
    for key in vgg_param_dict:
        print(key,vgg_param_dict[key][0].shape,vgg_param_dict[key][1].shape)
    print("vgg parameter loaded")
    return vgg_param_dict
    
vgg_param_dict = vgg_param_load(vgg16_npy_path) 

def get_conv_filter(name):
    return tf.constant(vgg_param_dict[name][0], name="filter")
    #so here load the weight for VGG-16, which is kernel, the kernel size for different covolution layers will show in function
    #vgg_param_load

def get_bias(name):
    return tf.constant(vgg_param_dict[name][1], name="biases")
    #here load the bias for VGG-16, the bias size will be 64,128,256,512,512, also shown in function vgg_param_load
  
def max_pool(inputs,name):
    with tf.variable_scope(name) as scope:
        value,index = tf.nn.max_pool_with_argmax(tf.to_double(inputs),ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=scope.name)
    return tf.to_float(value),index,inputs.get_shape().as_list()
    #here value is the max value, index is the corresponding index, the detail information is here 
    #https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/max_pool_with_argmax
    
    
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
    with tf.variable_scope(name) as scope:
        init = tf.constant_initializer(get_conv_filter(scope.name))
        filt = _variable_with_weight_decay('weights',shape = shape, initializer = init, wd = False,enc = True)
        tf.summary.histogram(scope.name+"weight",filt)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases_init = tf.constant_initializer(get_bias(scope.name))
        conv_biases = _variable_with_weight_decay('biases',shape = [shape[3]],initializer = conv_biases_init, wd= False, enc = True)
        tf.summary.histogram(scope.name+"bias",conv_biases)
        bias = tf.nn.bias_add(conv, conv_biases)
        conv_out = tf.nn.relu(batch_norm(bias,training_state,scope.name))
    return conv_out    



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
    with tf.variable_scope(name) as scope:
        filt = _variable_with_weight_decay('weights',shape = shape, initializer = _initialization(kernel_size,input_channel), wd = False,enc = False)
        tf.summary.histogram(scope.name+"weight",filt)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = _variable_with_weight_decay('biases',shape = out_channel, initializer = tf.constant_initializer(0.0), wd = False, enc = False)
        tf.summary.histogram(scope.name+"bias",conv_biases)        
        bias = tf.nn.bias_add(conv, conv_biases)
        conv_out = tf.nn.relu(batch_norm(bias,training_state,scope.name))      
    return conv_out
        #norm is used to identify if we should use batch normalization!
        

def batch_norm(bias_input, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(bias_input, is_training=False,center=False, reuse = True, scope=scope))
#is_training = Ture, it will accumulate the statistics of the movements into moving_mean and moving_variance. When it's 
#not in a training mode, then it would use the values of the moving_mean, and moving_variance. 
#shadow_variable = decay * shadow_variable + (1 - decay) * variable, shadow_variable, I think it's the accumulated moving
#average, and then variable is the average for this specific batch of data. For the training part, we need to set is_training
#to be True, but for the validation part, actually we should set it to be False!
                   
                   
def up_sampling(pool, ind, output_shape, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        #the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense, then the gradient is None, which will cut off the network. 
        #But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None. 
        #The usage for tf.scatter_nd is that: create a new tensor by applying sparse UPDATES(which is the pooling value) to individual values of slices within a 
        #zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_). If we ues the orignal code, the only thing we need to change is: changeing
        #from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),sparse_tensor) which will give us the gradients!!!
        ret = tf.reshape(ret, output_shape)
        return ret       



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
    
def _variable_on_cpu(name,shape,initializer,enc):
    """
    Help to create a variable
    Inputs: 
    name: corresponding layers name, conv1 or ...
    shape: the shape of the weight or bias
    initializer: The initializer from _ínitialization
    Outputs:
    Variable tensor
    """
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

