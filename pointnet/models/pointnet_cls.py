import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 8))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    # The first transform is for input
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=10) # output dimentions after fransform
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    # BxNx10
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    # BxNx10x1
    # the second parameter is for the output number of feature maps(FM) not the size of FM
    net = tf_util.conv2d(input_image, 64, [1,10],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    # BxNx1x64 ,in other words,the size of FM is Nx1 and 64 FMs
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    # BxNx1x64
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    # after squeeze BxNx64 ,and we know the shape of transform is Bx64x64, so we get the feature transform result BxNx64
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform) 
    # BxNx64
    net_transformed = tf.expand_dims(net_transformed, [2])
    
    point_feat = net_transformed # TODO record point-wise feature
    # BxNx1x64
    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    # BxNx1x64
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    # BxNx1x128
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    # BxNx1x1024

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    # Bx1x1x1024  Global Feature TODO
    global_feat = net
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    print(global_feat_expand)
    print(point_feat)
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)
    
#     net = tf.reshape(net, [batch_size, -1])
#     # Bx1024
#     net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
#                                   scope='fc1', bn_decay=bn_decay)
#     net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                           scope='dp1')
#     net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                   scope='fc2', bn_decay=bn_decay)
#     net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                           scope='dp2')
#     net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 32, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 3, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    print(net.shape)
    net = tf.squeeze(net, [2]) # BxNxC
    print(net.shape)
    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES
        label: B,
        for fluid pred BxNx3,label BxNx3
    """
    loss = tf.losses.mean_squared_error(label, pred)
    regression_loss = tf.reduce_sum(loss)
    print(regression_loss)
    tf.summary.scalar('classify loss', regression_loss)
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
#     classify_loss = tf.reduce_mean(loss)
#     tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)
    print(type(mat_diff_loss))
    return regression_loss + mat_diff_loss * reg_weight

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,8))
        outputs = get_model(inputs, tf.constant(True))
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print(outputs)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            variables = {gvar.op.name: value for gvar, value in zip(gvars, sess.run(gvars))}
            # print(variables)