# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
#%%
def batch_norm_liqi(x, name):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  name=name,
                                  variance_epsilon=epsilon)
    return x
def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
   
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
# conv_op函数用来创建卷积层并且把本层的参数存入参数列表
def conv_op(input_op, name, padding='SAME', kh=3, kw=3, n_out=1, dh=1, dw=1, data_dict=None, pretrain=False, bnflag=False, is_training=False):
    '''
    Args:
    input_op：输入的tensor
    name：这一层的名称
    kh：kernel height即卷积核的高
    kw：kernel weight即卷积核的宽
    n_out：卷积核数量即输出通道数
    dh：步长的高
    dw：步长的宽
    p：参数列表
    '''
    n_in = input_op.get_shape()[-1].value # 获取input_op的通道数

    with tf.name_scope(name) as scope: # 设置scope，生成的Variable使用默认的命名
        kernel = tf.get_variable(scope+"w",  # kernel（即卷积核参数）使用tf.get_variable创建
                                 shape=[kh, kw, n_in, n_out], # 【卷积核的高，卷积核的宽、输入通道数，输出通道数】
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d()) # 参数初始化
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32) # biases使用tf.constant赋值为0
        biases = tf.Variable(bias_init_val, trainable=True, name='b') # 将bias_init_val转成可训练的参数
        if pretrain:
            kernel = tf.assign(kernel,data_dict[name]['weights'])
            biases = tf.assign(biases,data_dict[name]['biases'])
        # 使用tf.nn.conv2d对input_op进行卷积处理，卷积核kernel，步长dh*dw，padding模式为SAME
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        z = tf.nn.bias_add(conv, biases) # 将卷积结果conv和bias相加
        return z # 将卷积层的输出activation作为函数结果返回

# 定义全连接层的创建函数
def fc_op(input_op, name, n_out, data_dict, pretrain=False, bnflag=False, is_training=False):  
    n_in = input_op.get_shape()[-1].value # 获取tensor的通道数

    with tf.name_scope(name) as scope:
        
        kernel = tf.get_variable(scope+"w", # 使用tf.get_variable创建全连接层的参数
                                 shape=[n_in, n_out], # 参数的维度有两个，输入通道数和输出通道数
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        # biases赋值0.1以避免dead neuron
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b') 

        if pretrain:
            kernel = tf.assign(kernel, data_dict[name]['weights'])
            biases = tf.assign(biases, data_dict[name]['biases']) 
        # 对输入变量input_op和kernel做矩阵乘法并加上biases。再做非线性变换activation
        activation = tf.nn.relu(tf.matmul(input_op, kernel) + biases, name=scope) 
        #if bnflag:
        #    z = batch_norm(z, n_out, is_training)

        out = tf.cond(is_training, lambda: tf.nn.dropout(activation, 0.4), lambda: activation) 
        #activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope) 
        #p += [kernel, biases]
        print name
        return activation

# 定义最大池化层的创建函数
def mpool_op(input_tensor=None, k=2, s=2, padding='SAME',name=None): 
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, k, k, 1], # 池化层尺寸kh*kw
                          strides=[1, s, s, 1], # 步长dh*dw
                          padding=padding,
                          name=name
                          )

def apool_op(input_op, name, kh, kw, dh, dw): 
    return tf.nn.avg_pool(input_op,
                          ksize=[1, kh, kw, 1], # 池化层尺寸kh*kw
                          strides=[1, dh, dw, 1], # 步长dh*dw
                          padding='VALID',
                          name=name)   

def lrn(input_op, name, radius, alpha, beta):
    return tf.nn.local_response_normalization(input_op,
                                                depth_radius=radius,
                                                alpha=alpha,
                                                beta=beta,
                                                bias=1.0,
                                                name=name)