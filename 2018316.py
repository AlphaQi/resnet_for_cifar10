"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2018 Netease,Inc.
Licensed under the MIT License (see LICENSE for details)
Written by LiQi
"""

import os
import sys
sys.path.append('./coco/PythonAPI')
import glob
import random
import math
import datetime
import json
import re
import logging
import numpy as np
import scipy.misc
import tensorflow as tf
import cifar10_input
import setool
############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, pretrain=False,  data_dict=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = setool.conv_op(input_op=input_tensor, name=conv_name_base + '2a',\
                       kh=1, kw=1,  n_out=nb_filter1,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2a')#//???????
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2b',\
                       kh=kernel_size, kw=kernel_size,  n_out=nb_filter2,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2b')#//???????
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2c',\
                       kh=1, kw=1,  n_out=nb_filter3,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2c')#//???????

    x = x+input_tensor
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, pretrain=False,strides=2, data_dict=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = setool.conv_op(input_op=input_tensor, name=conv_name_base + '2a',\
                       kh=1, kw=1,  dh=strides, dw=strides, n_out=nb_filter1,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2a')#//???????
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2b',\
                       kh=kernel_size, kw=kernel_size,  n_out=nb_filter2,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2b')#//???????
    x = tf.nn.relu(x)

    x = setool.conv_op(input_op=x, name=conv_name_base + '2c',\
                       kh=1, kw=1,  n_out=nb_filter3,data_dict=data_dict,
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name=bn_name_base + '2c')#//???????

    shortcut = setool.conv_op(input_op=input_tensor, name=conv_name_base + '1',\
                       kh=1, kw=1,  dh=strides, dw =strides,
                       n_out=nb_filter3,data_dict=data_dict,
                       pretrain=False)
    shortcut = setool.batch_norm_liqi(x=shortcut, name=bn_name_base + '1')#//???????
    x = x+shortcut
    x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
    return x

def resnet_graph(input_image, architecture, stage5=False, data_dict=None, pretrain=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = setool.conv_op(input_op=input_image, name='conv1',\
                       kh=7, kw=7,  dh=2, dw=2, n_out=64,data_dict=data_dict,#@
                       pretrain=False)
    x = setool.batch_norm_liqi(x=x, name='bn_conv1')#//???????
    x = tf.nn.relu(x)

    C1 = x = setool.mpool_op(input_op=x, kh=3,kw=3, dh=2, dw=2)#@
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1,
                    pretrain=False,
                    data_dict=data_dict)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',
                    pretrain=False,
                    data_dict=data_dict)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',
                    pretrain=False,
                    data_dict=data_dict)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',#@
                    pretrain=False,
                    data_dict=data_dict)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',
                    pretrain=False,
                    data_dict=data_dict)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',
                    pretrain=False,
                    data_dict=data_dict)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',
                    pretrain=False,
                    data_dict=data_dict)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',#@
                    pretrain=False,
                    data_dict=data_dict)
    block_count = {"resnet50": 1, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i),
                    pretrain=False,
                    data_dict=data_dict)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',#@
                    pretrain=False,
                    data_dict=data_dict)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',
                    pretrain=False,
                    data_dict=data_dict)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',
                    pretrain=False,
                    data_dict=data_dict)
    else:
        C5 = None
    #if image.shape = 32, cg = [batchsize , 4,4,2048]

    return C5

def loss(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss

class cifar():

    def __init__(self, model_dir, mode="training" , config=None):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        # self.config = config
        self.model_dir = model_dir
        self.BATCH_SIZE = 128
        self.learning_rate = 0.01
        self.MAX_STEP = 100000 # with this setting, it took less than 30 mins on my laptop to train.

    def build(self, mode, config, images):

        assert mode in ['training', 'inference']
        # Image size must be dividable by 2 multiple times
        # h, w = config.IMAGE_SHAPE[:2]
        # if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        #     raise Exception("Image size must be dividable by 2 at least 6 times "
        #                     "to avoid fractions when downscaling and upscaling."
        #                     "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # input_image = tf.placeholder(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        C5 = resnet_graph(images, "resnet50", stage5=True)
        x = setool.conv_op(input_op=C5, name='lqa',\
                       kh=4, kw=4,  n_out=512,padding="VALID")
        x = tf.nn.relu(x)
        x = setool.conv_op(input_op=x, name='lqb',\
                       kh=1, kw=1,  n_out=10)
        logits = x
        softmaxOut = tf.nn.softmax(logits, name="softmaxOut")
        return softmaxOut, logits

    def train(self):
        data_dir = "./data/"
        log_dir = "./logs/"

        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
                                                    is_train=True,
                                                    batch_size=self.BATCH_SIZE,
                                                    shuffle=True)

        softmaxOut, logits = self.build(images=images, mode="training",config=None)
        loss_softmax_cross = loss(logits, labels)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss_softmax_cross,  global_step=my_global_step)
        saver = tf.train.Saver(tf.global_variables())
        #tensorboard
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #tensorboard
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        try:
            for step in np.arange(self.MAX_STEP):
                if coord.should_stop():
                        break
                _, loss_value = sess.run([train_op, loss_softmax_cross])
                   
                if step % 5 == 0:                 
                    print ('Step: %d, loss: %.4f' % (step, loss_value))
                    
                if step % 100 == 0:
                    #tensorboard
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)                
        
                if step % 2000 == 0 or (step + 1) == self.MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    liqi = cifar(model_dir="./logs")
    liqi.train()