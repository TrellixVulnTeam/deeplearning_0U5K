#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from nose.tools import assert_equal
import numpy as np
import tensorflow as tf


from config import cfg


small_addon_for_BCE = 1e-6


class MiddleAndReg(object):
    
    def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        self.input = input
        self.training = training
        self.y = tf.placeholder(tf.float32, [None, 3])

        with tf.variable_scope('MiddleAndReg_' + name):

            # 3D reduction
            # input shape (B, 40, 40, 50, 128)
            temp_conv = ConvMD(3, 128, 64, 3, (2, 2, 2),
                               (1, 1, 1), self.input, name='conv1')
            # (B, 20, 20, 25, 64)
            temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 2),
                               (0, 1, 0), temp_conv, name='conv2')
            # (B, 18, 20, 12, 64)
            temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1),
                               (0, 1, 1), temp_conv, name='conv3')
            # (B, 8, 20, 12, 64)
            # print('after conv3:', temp_conv)
            temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
            # print('perm transpose: ', temp_conv)
            width = temp_conv.get_shape()[1].value
            depth = temp_conv.get_shape()[2].value
            temp_conv = tf.reshape(temp_conv, [-1, width * depth, 64, 8])
            # print('after reshape: ', temp_conv)

            temp_conv = tf.reduce_mean(temp_conv, axis=3, keepdims=True)  # todo reduce_max@3
            # print('after reduce_mean:', temp_conv)
            temp_conv = tf.reshape(temp_conv, [-1, width, depth, 64])
            # print(temp_conv.shape)  # (B, 20, 12, 64)

            # 2D reduction
            temp_conv = ConvMD(2, 64, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv4')
            # (B, 10, 6, 128 )
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv5')
            # down_sampling1 = tf.layers.average_pooling2d(temp_conv, 2, 2, name="down1")
            down_sampling1 = ConvMD(2, 128, 128, 2, (2, 2), (0, 0),
                               temp_conv, training=self.training, name='down1')

            # print('down_sampling1', down_sampling1)

            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv6')
            down_sampling1 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling1, training=self.training, name='down1_conv1')
            # down_sampling2 = tf.layers.average_pooling2d(temp_conv, 2, 2, name="down2")
            down_sampling2 = ConvMD(2, 128, 128, 2, (2, 2), (0, 0),
                                    temp_conv, training=self.training, name='down2')


            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv7')
            down_sampling1 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling1, training=self.training, name='down1_con2')
            down_sampling2 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling2, training=self.training, name='down2_conv1')
            # down_sampling3 = tf.layers.average_pooling2d(temp_conv, 2, 2, name="down3")
            down_sampling3 = ConvMD(2, 128, 128, 2, (2, 2), (0, 0),
                                    temp_conv, training=self.training, name='down3')

            # down_sampling1 = tf.layers.conv2d_transpose(down_sampling1, 128, 3, 2, padding='valid',
            #                                             name='deconv1')
            # down_sampling2 = tf.layers.conv2d_transpose(down_sampling2, 128, 3, 2, padding='valid',
            #                                             name='deconv2')
            # down_sampling3 = tf.layers.conv2d_transpose(down_sampling3, 128, 3, 2, padding='valid',
            #                                             name='deconv3')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv8')
            down_sampling1 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling1, training=self.training, name='down1_con3')
            down_sampling2 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling2, training=self.training, name='down2_conv2')
            down_sampling3 = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                                    down_sampling3, training=self.training, name='down3_conv1')

            # down_sampling1 = Deconv2D(128, 128, 2, 2, (0, 0), down_sampling1, training=self.training,
            #                           name='deconv1')
            # print('down_sampling1 after deconv', down_sampling1)
            # down_sampling2 = Deconv2D(128, 128, 2, 2, (0, 0), down_sampling2, training=self.training,
            #                           name='deconv2')
            # down_sampling3 = Deconv2D(128, 128, 2, 2, (0, 0), down_sampling3, training=self.training,
            #                           name='deconv3')
            temp_conv = ConvMD(2, 128, 128, 2, (2, 2), (0, 0),
                                    temp_conv, training=self.training, name='down4')

            print('down_sampling1: ', down_sampling1)
            temp_conv = tf.concat([temp_conv, down_sampling1, down_sampling2, down_sampling3], axis=3)
            print('temp_conv: ', temp_conv)  # (B, 5, 3, 512)
            cin = temp_conv.get_shape()[-1]
            r_map = ConvMD(2, cin, 192,  3, (1, 1), (1, 1),
                           temp_conv, training=self.training, activation=False, bn=False, name='conv9')
            r_map = ConvMD(2, 192, 48, 3, (1, 1), (1, 1),
                           r_map, training=self.training, activation=False, bn=False, name='conv10')
            print('r_map shape:', r_map)
            height = r_map.get_shape()[1]
            width = r_map.get_shape()[2]
            fms = r_map.get_shape()[3]
            final_features = tf.reshape(r_map, [-1, height * width * fms])
            middle1 = 180
            middle2 = 45
            print('final_features shape:', final_features.shape)
            final_features_middle1 = tf.layers.dense(final_features, middle1, activation=tf.nn.relu, name="fcn1")
            final_features_middle2 = tf.layers.dense(final_features_middle1, middle2, activation=tf.nn.relu, name="fcn2")
            self.pred = tf.layers.dense(final_features_middle2, 3, name="output")
            print("pred:\n", self.pred)
            print("********************************")
            # FIXME
            self.regression_loss = tf.losses.mean_squared_error(self.y, self.pred)
            print('regression_loss', self.regression_loss)
            self.loss = self.regression_loss
            print(self.loss)


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def ConvMD(M, Cin, Cout, k, s, p, input, training=True, activation=True, bn=True, name='conv'):
    """ConvMD(3, 128, 64, 3, (2, 1, 1),
                               (1, 1, 1), self.input, name='conv1')
    :param M: dimention
    :param Cin: input channels
    :param Cout: output channels
    :param k: kernel size
    :param s: stride
    :param p: padding # why padding manual
    :param input:
    :param training:
    :param activation:
    :param bn:
    :param name:
    :return:
    """
    temp_p = np.array(p)
    # ccx (1, 1):Number of values padded to the edges of each axis.
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    # if M=2 [0, 1, 1, 0] else M=3 [0, 1, 1, 1, 0]

    with tf.variable_scope(name) as scope:
        if(M == 2):
            # ccx repeat element by element
            paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)  # [[0, 0], [1, 1], [1, 1], [0, 0]]
            pad = tf.pad(input, paddings, "CONSTANT")  # default 0
            temp_conv = tf.layers.conv2d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope)
        if(M == 3):
            paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv3d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(temp_conv, axis=-1, fused=True,
                                                      training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv


def Deconv2D(Cin, Cout, k, s, p, input, training=True, bn=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        # change paddding from 'same' to 'valid'
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="same", reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)


if(__name__ == "__main__"):
    m = MiddleAndReg(tf.placeholder(
        tf.float32, [None, cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT, cfg.INPUT_DEPTH, 128]))
    tf.estimator.DNNClassifier.export_savedmodel()