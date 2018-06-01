#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
from nose.tools import assert_equal
import tensorflow as tf

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, batch_size, k_dynamics, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        count = 0
        result = []
        for screen in range(batch_size):
            pointwise = self.batch_norm.apply(self.dense.apply(inputs[count: count + k_dynamics[screen]]), training)

            # n [K, 1, units]( TODO haha,just like the max polling of conv2d)
            aggregated = tf.reduce_max(pointwise, axis=1, keepdims=True)  # pooling points in a voxel
            """
            change all above to conv2D
            """
            # [K, T, units]
            repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

            # [K, T, 2 * units]
            concatenated = tf.concat([pointwise, repeated], axis=2)

            result.append(concatenated)

            count += k_dynamics[screen]

        concatenated_all_batch = tf.concat(result, axis=0)

        # TODO pointwise + aggregated(global features) the same as the pointnet
        mask = tf.tile(mask, [1, 1, 2 * self.units]) # ccx (ΣK, T, output_channels) here
        # TODO use shared mlp,in other means ,expand input as [K, T, 2 * units, 1] conv2d [1, 2 * units, 1, 2 * units]
        concatenated_all_batch = tf.multiply(concatenated_all_batch, tf.cast(mask, tf.float32)) # ccx item corresponded
        # [K, T, out_channels]
        return concatenated_all_batch # ccx (ΣK, T, output_channels)


class PIL(object):
    def __init__(self):
        pass


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]  ccx: all files in a batch, so we get ΣK.
        assert_equal(cfg.VOXEL_POINT_FEATURE, 11)

        self.part_feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, cfg.VOXEL_PART_FEATURE], name='feature') # feature dimention can be set a variable
        # [ΣK]
        # TODO centoid particle here
        self.centroid = tf.placeholder(tf.float32, shape=(None, 3), name='centroid')
        self.k_dynamics = tf.placeholder(tf.int32, shape=(None), name="k_dynamics")
        # self.k_dynamics = tf.cast(self.k_dynamics, tf.int32)
        print(self.k_dynamics[0])
        concat_feature = []
        count = 0
        # if self.k_dynamics.shape[0] < self.batch_size:
        #    self.batch_size = self.k_dynamics.shape[0]
        # print("self.batch_size:\n", self.batch_size)
        for screen in range(self.batch_size):
            num = self.k_dynamics[screen]
            concat_feature.append(tf.concat([self.part_feature[count: count + num],
                                             self.part_feature[count: count + num,:,:3]-
                                             self.centroid[screen]],axis=2))
            # if (count + num) == self.part_feature.shape[0]:
            #   print("break")
            #    break
            count += num


        # self.feature = tf.concat([self.part_feature, self.part_feature[:,:,:3]-self.centroid],axis=2)
        self.feature = tf.concat(concat_feature, axis=0)

        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w) ccx: the second dimention of tensor each row stores (iswhich, d, h, w) iswhich tag the different file in a batch
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')
        # self.seg_single_feature = tf.scatter_nd(self.coordinate, self.part_feature)


        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')

        # boolean mask [K, T, 2 * units] # FIXME: ccx fatal  from different file we get different K.
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keepdims=True), 0)   # (ΣK, T, 1) here, keepdims is true means keep the dimentions but the length only 1

        x = self.vfe1.apply(self.feature, mask, self.batch_size, self.k_dynamics, self.training)
        x = self.vfe2.apply(x, mask, self.batch_size, self.k_dynamics, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)
        # ccx: D' x H' x W' x  C, where C is dimention of voxelwise feature.
        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        # FIXME ccx AWESOME- a sparse tensor
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, cfg.INPUT_WIDTH,  cfg.INPUT_HEIGHT, cfg.INPUT_DEPTH, 128])
        """
        scatter_nd()
        
        
        
        
        
        
        
        
        
        
        
        
        incidices, updates, shape
        return a tensor
        
        locate voxel by fileid and voxel_index, the others padding 0, so a 5-D tensor with shape [2, width, height, depth, 128] returned
        
        """




