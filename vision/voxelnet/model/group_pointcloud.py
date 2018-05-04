#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

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

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.v(inputs), training)

        #n [K, 1, units]( TODO haha,just like the max polling of conv2d)
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True) # pooling points in voxel
        """
        change all above to conv2D
        """
        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)
        # TODO pointwise + aggregated(global features) the same as the pointnet
        mask = tf.tile(mask, [1, 1, 2 * self.units])
        # TODO use shared mlp,in other means ,expand input as [K, T, 2 * units, 1] conv2d [1, 2 * units, 1, 2 * units]
        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32)) # ccx item corresponded
        # [K, T, out_channels]
        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]  ccx: all files in a batch, so we get ΣK.
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w) ccx: the second dimention of tensor each row stores (iswhich, d, h, w) iswhich tag the different file in a batch
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')

        # boolean mask [K, T, 2 * units] # FIXME: ccx fatal  from different file we get different K.
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keep_dims=True), 0)   # (ΣK, T, 1) now
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)
        # ccx: D' x H' x W' x  C, where C is dimention of voxelwise feature.
        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        # FIXME ccx AWESOME- a sparse tensor
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])
        """
        scatter_nd()
        incidices, updates, shape
        return a tensor
        """



