# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:18:45 2017

@author: s161488
"""

import tensorflow as tf
import numpy as np


def unpool_with_argmax(pool, ind, shape_ori, name = None, ksize=[1, 2, 2, 1]):

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
        output_shape = shape_ori
        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        #first let's create batch matrix, the batch_range determine the batch index for each element in the matrix(0,to,batch_size-1),
        #the size will be as same as the size of output from maxpooling! And also when we reshape the output(if the batch size is not 1), then the order of
        #output  is first read from the first batch, after finished we start to read the second batch, so this way is exactly does the same thing, the 
        #concat index for the first batch are all zero, then the concat index for the second batch are all one, so this upsampling way is really correct!! 
        #actually the flattened pooling index represent the index for the pooling value, which is in the range of (0-totalnumber of pixels)!!
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret
    
def max_pool(inputs,name):
    value,index = tf.nn.max_pool_with_argmax(tf.to_double(inputs),ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    print('value shape',value.shape)
    print('index shape',index.shape)
    return tf.to_float(value),index,inputs.get_shape().as_list()
    
def Test_Upsampling():
    gen_array = np.array([[[[0,1,3],
                           [1,3,6],
                           [2,6,7],
                           [3,7,2],
                           [5,5,5]],

                           [[7,4,1],
                            [8,9,9],
                            [5,8,2],
                            [3,5,8],
                            [6,3,9]],

                           [[2,6,2],
                            [4,3,7],
                            [3,2,3],
                            [9,5,5],
                            [9,4,4]],
                            
                           [[7,3,1],
                            [4,7,4],
                            [6,5,6],
                            [7,8,9],
                            [2,9,2]]]])
                            
                
    xplaceholder = tf.placeholder(tf.float32,np.shape(gen_array))
    value,index,oridex = max_pool(xplaceholder,'test')
    sp_dense = unpool_with_argmax(value,index,np.shape(gen_array),name='test',ksize=[1,2,2,1])
    print(np.shape(gen_array))
    print(sp_dense)
    with tf.Session() as sess:
        feed_dict = {
        xplaceholder:gen_array}    
        fetches = [value,index,sp_dense]
        maxv,maxi,spdense = sess.run(fetches,feed_dict)
    return maxv,maxi,spdense



def Test_Gradient():
    indices = tf.placeholder(tf.int64,(None,2))
    values = tf.placeholder(tf.float32,(None,))
    sparse_tensor = tf.SparseTensor(indices,values,(5,7))
    dense_tensor1 = tf.sparse_tensor_to_dense(sparse_tensor)
    dense_tensor2 = tf.scatter_nd(indices,values,shape=[5,7])
    dense_tensor3 = tf.sparse_add(tf.zeros((5,7)),sparse_tensor)
    sum1 = tf.reduce_sum(dense_tensor1)
    sum2 = tf.reduce_sum(dense_tensor2)
    sum3 = tf.reduce_sum(dense_tensor3)
    
    print('dense_tensor1',tf.gradients(sum1,values)) #None
    print('dense_tensor2',tf.gradients(sum2,values)) #tf.Tensor 'gradients_1/ScatterNd_grad/GatherNd:0'
    print('dense_tensor3',tf.gradients(sum3,values)) #tf.Tensor 'gradients_2/SparseTensorDenseAdd_grad/GatherNd:0'
    
    
#gen_array = np.array([[[[0,1,3],
#                           [1,3,6],
#                           [2,6,7],
#                           [3,7,2],
#                           [5,5,5],
#                           [4,2,4]],
#
#                           [[7,4,1],
#                            [8,9,9],
#                            [5,8,2],
#                            [3,5,8],
#                            [6,3,9],
#                            [7,7,9]],
#
#                           [[2,6,2],
#                            [4,3,7],
#                            [3,2,3],
#                            [9,5,5],
#                            [9,4,4],
#                            [8,2,6]],
#                            
#                           [[7,3,1],
#                            [4,7,4],
#                            [6,5,6],
#                            [7,8,9],
#                            [2,9,2],
#                            [5,1,5]]]])
#    
