#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import os
import sys
import time

import tensorflow as tf
import cv2
from numba import jit

from config import cfg
from utils import *
from model.group_pointcloud import FeatureNet
from model.rpn import MiddleAndReg


class RPN3D(object):

    def __init__(self,
                 cls='Car',
                 single_batch_size=2,  # batch_size_per_gpu
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 alpha=1.5,
                 beta=1,
                 avail_gpus=['0']):
        # hyper parameters and status
        self.scope = []
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.beta = beta
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [ self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01 ]
        lr = tf.train.piecewise_constant(self.epoch, boundaries, values)

        # build graph
        # input placeholders
        self.is_train = tf.placeholder(tf.bool, name='phase')

        self.vox_feature = []
        self.vox_number = []
        self.vox_coordinate = []
        self.labels = []
        self.vox_centroid = []
        self.vox_k_dynamic = []
        self.targets = []
        self.screen_size = []
        self.outputs = []
        self.concat_feature = []
        self.voxelwise = None  # 郑重申明：获取voxelwise仅仅作为中间值在session中定义ScatterND无需feed value

        self.opt = tf.train.AdamOptimizer(lr)
        self.final_feature = []
        self.pred = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)) as scope:
                    self.scope.append(scope)
                    # must use name scope here since we do not want to create new variables
                    # graph
                    feature = FeatureNet(
                        training=self.is_train, batch_size=self.single_batch_size)
                    rpn = MiddleAndReg(
                        input=feature.outputs, alpha=self.alpha, beta=self.beta, training=self.is_train)
                    tf.get_variable_scope().reuse_variables()
                    # input
                    self.vox_feature.append(feature.part_feature)
                    self.vox_number.append(feature.number)
                    self.vox_coordinate.append(feature.coordinate)
                    self.vox_centroid.append(feature.centroid)
                    self.vox_k_dynamic.append(feature.k_dynamics)
                    self.labels.append(rpn.y)
                    self.screen_size.append(feature.screen_size)
                    self.final_feature.append(feature.feature)
                    self.voxelwise = feature.voxelwise
                    # output
                    feature_output = feature.outputs
                    self.outputs.append(feature_output)
                    self.pred.append(rpn.pred)
                    # loss and grad
                    if idx == 0:
                        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    self.loss = rpn.loss
                    self.params = tf.trainable_variables()

        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # loss and optimizer
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.optimazer = self.opt.minimize(self.loss)
        #     self.grads = average_gradients(self.tower_grads)
        #     self.update = [self.opt.apply_gradients(
        #         zip(self.grads, self.params), global_step=self.global_step)]
        #     self.gradient_norm = tf.group(*self.gradient_norm)
        #
        # self.update.extend(self.extra_update_ops)
        # self.update = tf.group(*self.update)

        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

    def train_step(self, session, data, train=False, summary=False):
        # input:
        #     (N) tag
        #     (N, N') label
        #     vox_feature
        #     vox_number
        #     vox_coordinate

        labels = data[0]
        print('labels data:\n', labels, '\n')
        vox_feature = data[1]
        vox_number = data[2]
        vox_coordinate = data[3]
        vox_centroid = data[4]
        vox_k_dynamic = data[5]

        input_feed = {}
        input_feed[self.is_train] = True
        for idx in range(len(self.avail_gpus)):
            input_feed[self.screen_size[idx]] = self.single_batch_size
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_centroid[idx]] = vox_centroid[idx]
            input_feed[self.vox_k_dynamic[idx]] = vox_k_dynamic[idx]

            # session.graph.add_to_collection('concat_feature', concat_feature)
            start = time.clock()
            final_feature_eval = self.concat_feature[idx].eval(session=session, feed_dict=input_feed)
            print('concat feature calculate', time.clock() - start)
            input_feed[self.final_feature[idx]] = final_feature_eval
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.labels[idx]] = labels[idx]

        if train:
            output_feed = [self.loss, self.optimazer, self.pred[0]]
        else:
            output_feed = [self.loss]
        if summary:
            output_feed.append(self.train_summary)
        # TODO: multi-gpu support for test and predict step
        return session.run(output_feed, input_feed)


def average_gradients(tower_grads):
    # ref:
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    pass
