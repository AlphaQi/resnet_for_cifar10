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
import setool
import config
import utils
import argparse
import coco_data_input

GLOBAL_BATCH_SIZE = 2
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

    x = setool.mpool_op(input_tensor=x, k=3, s=2)#@
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

    return C2, C3, C4, C5

def loss(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)
        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss


class RPN_net():
    def __init__(self, in_feature=None, anchors_per_location=3, anchor_stride=1):
        self.in_feature = in_feature
        self.anchors_per_location = anchors_per_location
        self.strides = anchor_stride

    def build_rpn_model(self):
        shared = setool.conv_op(input_op=self.in_feature, name='rpn_conv_shared', \
                                dh=self.strides, dw=self.strides, n_out=512)
        shared = tf.nn.relu(shared)

        #classfication
        x = setool.conv_op(input_op=shared, name='rpn_class_raw', n_out=2*self.anchors_per_location,\
                            kh=1, kw=1, padding='VALID')
        rpn_class_logits = tf.reshape(x, [GLOBAL_BATCH_SIZE, -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")

        #regression
        x = setool.conv_op(input_op=shared, name='rpn_bbox_pred', n_out=4*self.anchors_per_location, \
                            kh=1, kw=1, padding='VALID')
        rpn_bbox = tf.reshape(x, [GLOBAL_BATCH_SIZE, -1, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox]

############################################################
#  Loss Functions
############################################################
def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify, like flatten
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class,
                                                        logits=rpn_class_logits)

    # if tf.size(loss) > tf.size([0]):
    #     loss = tf.mean(loss)
    # else:
    #     loss = tf.constant(0.0)
    result = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))  
    # loss = tf.switch(tf.size(loss) > 0, tf.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    #target_bbox shiji 256*4, zheng yangben, rpn_match shiji fenlei label
    # rpn_bbox, predict 261888*4
    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    #I changed here ,add two lines code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    target_bbox = tf.cast(target_bbox, "float32") 
    rpn_bbox = tf.cast(rpn_bbox, "float32") 

    diff = tf.abs(target_bbox - rpn_bbox)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    result = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))  
    # loss = tf.switch(tf.size(loss) > 0, tf.mean(loss), tf.constant(0.0))
    return loss


class MaskRCNN():

    def __init__(self, model_dir, mode="training" , config=None):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.BATCH_SIZE = GLOBAL_BATCH_SIZE
        self.learning_rate = 0.0001
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

        C2, C3, C4, C5 = resnet_graph(images, "resnet50", stage5=True)

        #128*4*4*256
        P5 = setool.conv_op(input_op=C5, name='fpn_c5p5',kh=1, kw=1,  n_out=256)
        P4 = setool.conv_op(input_op=C4, name='fpn_c4p4',kh=1, kw=1,  n_out=256) + \
                     tf.image.resize_images(P5, [64,64])
        P3= setool.conv_op(input_op=C3, name='fpn_c3p3',kh=1, kw=1,  n_out=256) + \
                     tf.image.resize_images(P4, [128, 128])
        P2= setool.conv_op(input_op=C2, name='fpn_c2p2',kh=1, kw=1,  n_out=256) + \
                     tf.image.resize_images(P3, [256, 256])


        P2 = setool.conv_op(input_op=P2, name='fpn_p2',n_out=256)
        P3 = setool.conv_op(input_op=P3, name='fpn_p3',n_out=256)
        P4 = setool.conv_op(input_op=P4, name='fpn_p4',n_out=256)
        P5 = setool.conv_op(input_op=P5, name='fpn_p5',n_out=256)
        P6 = setool.mpool_op(input_tensor=P5, k=1, s=2, name="fpn_p6")


        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        # Generate Anchors
        self.anchors = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                      self.config.RPN_ANCHOR_RATIOS,
                                                      self.config.BACKBONE_SHAPES,
                                                      self.config.BACKBONE_STRIDES,
                                                      self.config.RPN_ANCHOR_STRIDE)
        #(32, 64, 128, 256, 512)  3, [256,128,64,32,16], [4, 8, 16, 32, 64], 1
        rpn_P6 = RPN_net(P6, anchor_stride = self.config.RPN_ANCHOR_STRIDE).build_rpn_model()
        rpn_P5 = RPN_net(P5, anchor_stride = self.config.RPN_ANCHOR_STRIDE).build_rpn_model()
        rpn_P4 = RPN_net(P4, anchor_stride = self.config.RPN_ANCHOR_STRIDE).build_rpn_model()
        rpn_P3 = RPN_net(P3, anchor_stride = self.config.RPN_ANCHOR_STRIDE).build_rpn_model()
        rpn_P2 = RPN_net(P2, anchor_stride = self.config.RPN_ANCHOR_STRIDE).build_rpn_model()

        rpn_class_logits = tf.concat([rpn_P2[0], rpn_P3[0], rpn_P4[0], rpn_P5[0], rpn_P6[0]], 1)
        rpn_class = tf.concat([rpn_P2[1], rpn_P3[1], rpn_P4[1], rpn_P5[1], rpn_P6[1]], 1)
        rpn_bbox = tf.concat([rpn_P2[2], rpn_P3[2], rpn_P4[2], rpn_P5[2], rpn_P6[2]], 1)
        # print(rpn_class_logits.shape)
        # print(rpn_class.shape)
        # print(rpn_bbox.shape)

        return rpn_class_logits, rpn_bbox, rpn_class
        # sys.exit(0)
    
        # x = setool.conv_op(input_op=C5, name='lqa',\
        #                kh=4, kw=4,  n_out=512,padding="VALID")
        # x = tf.nn.relu(x)
        # x = setool.conv_op(input_op=x, name='lqb',\
        #                kh=1, kw=1,  n_out=10)
        # logits = x
        # softmaxOut = tf.nn.softmax(logits, name="softmaxOut")

    def train(self, train_dataset=None):
        # data_dir = "./data/"
        log_dir = "./logs/"

        # images, labels = cifar10_input.read_cifar10(data_dir=data_dir,
        #                                             is_train=True,
        # #                                             batch_size=self.BATCH_SIZE,
        #                                             shuffle=True)

        train_generator = coco_data_input.data_generator(train_dataset, self.config, shuffle=True,
                                                        batch_size=self.config.BATCH_SIZE)

        #if use BATCH_SIZE instead of None, it can print the size
        images = tf.placeholder(tf.float32, [None, 1024, 1024, 3])
        input_rpn_classfi = tf.placeholder(tf.float32, [None, 261888, 1])
        input_rpn_bbox = tf.placeholder(tf.float32, [None, 256, 4])

        rpn_class_logits, rpn_bbox, rpn_softmaxOut = self.build(images=images, mode="training",config=None)

        loss_softmax_cross = rpn_class_loss_graph(input_rpn_classfi, rpn_class_logits)
        loss_bbox_regression = rpn_bbox_loss_graph(self.config, input_rpn_bbox, input_rpn_classfi, rpn_bbox)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        soft_op = optimizer.minimize(loss_softmax_cross,  global_step=my_global_step)
        bbox_op = optimizer.minimize(loss_bbox_regression,  global_step=my_global_step)
        saver = tf.train.Saver(tf.global_variables())
        #tensorboard
        # summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #tensorboard
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        try:
            for step in np.arange(self.MAX_STEP):
                if coord.should_stop():
                        break

                inputs, _ = next(train_generator)
                _, _, class_loss, regre_loss = sess.run([soft_op, bbox_op, loss_softmax_cross, loss_bbox_regression],\
                                feed_dict={images: inputs[0], input_rpn_classfi: inputs[1], input_rpn_bbox: inputs[2]})
                
                if step % 5 == 0:                 
                    print ('Step: %d' % (step))
                    print ('class_loss  : ', class_loss.mean())
                    print ('regre_loss  : ', regre_loss.mean())
                    print ('class_loss shape  : ', class_loss.shape)
                    print ('regre_loss shape : ',regre_loss.shape)
                    
                # if step % 100 == 0:
                #     #tensorboard
                #     summary_str = sess.run(summary_op)
                #     summary_writer.add_summary(summary_str, step)                
        
                if step % 50 == 0 or (step + 1) == self.MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    config = config.Config()
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')

    args = parser.parse_args()
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = coco_data_input.CocoDataset()
        dataset_train.load_coco(args.dataset, "val", year=2014)
        # dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()
    liqi = MaskRCNN(model_dir="./logs", config=config)
    liqi.train(train_dataset=dataset_train)