#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
from nose.tools import assert_equal
import tensorflow as tf

from config import cfg
# from utils import tf_util


class PILayer(object):
    def __init__(self, out_channels, name):
        self.units = int(out_channels / 2)
        self.name = name

    def apply(self, inputs, mask, training):
        num_point = inputs.get_shape()[1].value
        feature_dims = inputs.get_shape()[2].value  # feature dimention
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            self.conv2d = tf.layers.Conv2D(self.units, [1, feature_dims], strides=[1, 1], padding='valid',
                                           activation=tf.nn.relu, _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)
        input_expend = tf.expand_dims(inputs, -1)
        # (ΣK, T, 14, 1)

        pointwise = self.batch_norm(self.conv2d.apply(input_expend), training)
        # print('pointwise after conv2d and bn: ', pointwise)
        # (ΣK, T, 1, self.units)
        pointwise = tf.reshape(pointwise, [-1, num_point, self.units])
        # print('pointwise after reshape: ', pointwise)
        # (ΣK, T, self.units)
        aggregated = tf.reduce_mean(pointwise, axis=1, keepdims=True)  # todo reduce_max@1
        # (ΣK, 1, self.units)
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])
        # (ΣK, T, self.units)
        concatenated = tf.concat([pointwise, repeated], axis=2)
        # (ΣK, T, out_channels)

        # (ΣK, T, 1) -> (ΣK, T, 2 * units)
        mask = tf.tile(mask, [1, 1, 2 * self.units])
        # clear t which is all 0 in 3D grid
        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32)) 
        # (K, T, out_channels)
        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        # FIXME: ccx fatal  from different file we get different K.
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]  ccx: all files in a batch, so we get ΣK.
        assert_equal(cfg.VOXEL_POINT_FEATURE, 11)

        # feature dimention can be set a variable
        self.part_feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, cfg.VOXEL_PART_FEATURE], name='part_feature')
        # [ΣK]
        # TODO centoid particle here merge pos and vel
        self.centroid = tf.placeholder(tf.float32, shape=(None, 6), name='centroid')
        # self.centroid_pos = tf.placeholder(tf.float32, shape=(None, 3), name='centroid_pos')
        # self.centroid_vel = tf.placeholder(tf.float32, shape=(None, 3), name='centroid_vel')
        self.k_dynamics = tf.placeholder(tf.int32, shape=(None), name="k_dynamics")
        # self.k_dynamics = tf.cast(self.k_dynamics, tf.int32)
        print(self.k_dynamics[0])

        # (ΣK, T, 1) here use time:boolean mask [K, T, 2 * units],
        # keepdims is true means keep the dimentions but the length only 1
        mask = tf.not_equal(tf.reduce_max(
            self.part_feature, axis=2, keepdims=True), 0)
        
        self.screen_size = tf.placeholder(tf.int32, name="screen_size")
        # feed by concat feature in train_step   add \Delta v, so feature dimention from 11 to 14
        self.feature = tf.placeholder(tf.float32, shape=(None, cfg.VOXEL_POINT_COUNT, 14), name="feature")
        # TODO field partition
        # self.feature = tf.concat([self.part_feature, self.part_feature[:,:,:3]-self.centroid],axis=2)
        # self.feature = tf.concat(concat_feature, axis=0)

        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, h, d, w)
        # the second dimention of tensor each row stores (iswhich, d, h, w) iswhich tag the different file in a batch
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')
        # self.seg_single_feature = tf.scatter_nd(self.coordinate, self.part_feature)

        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        #     self.vfe1 = VFELayer(32, 'VFE-1')
        #     self.vfe2 = VFELayer(128, 'VFE-2')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.pil1 = PILayer(32, 'PIL-1')
            self.pil2 = PILayer(128, 'PIL-2')

        # x = self.vfe1.apply(self.feature, mask, self.batch_size, self.k_dynamics, self.training)
        # x = self.vfe2.apply(x, mask, self.batch_size, self.k_dynamics, self.training)


        # all_fields = []
        # for i in range(4):
        #     field_feature = self.feature[:, :, i*3:i*3+3]
        #     field_feature = self.pil1.apply(field_feature, mask, self.training)
        #     field_feature = self.pil2.apply(field_feature, mask, self.training)
        #     all_fields.append(field_feature)

        x = self.pil1.apply(self.feature, mask, self.training)
        x = self.pil2.apply(x, mask, self.training)

        # x = tf.concat(all_fields, axis=2)
        # [ΣK, 128]
        self.voxelwise = tf.reduce_mean(x, axis=1)  # reduce_max@2
        print(self.voxelwise)
        # ccx: D' x H' x W' x  C, where C is dimention of voxelwise feature.
        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        # FIXME ccx AWESOME- a sparse tensor

        # self.outputs = tf.placeholder(tf.float32, shape=(None, cfg.INPUT_WIDTH,  cfg.INPUT_HEIGHT,
        #                                                  cfg.INPUT_DEPTH, 128), name="scatter_nd_holder")
        self.outputs = tf.scatter_nd(
            self.coordinate, self.voxelwise, [self.batch_size, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, cfg.INPUT_DEPTH, 128])
        print(self.outputs)
        """
        scatter_nd()       
        incidices, updates, shape
        return a tensor
        locate voxel by fileid and voxel_index, the others padding 0, 
            so a 5-D tensor with shape [2, width, height, depth, 128] returned
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [-1, cfg.INPUT_WIDTH,  cfg.INPUT_HEIGHT, cfg.INPUT_DEPTH, 128])
        """


if __name__ == "__main__":
    feature = FeatureNet(True, 3)

