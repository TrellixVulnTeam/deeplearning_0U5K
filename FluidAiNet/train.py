#!/usr/bin/env python
# -*- coding:UTF-8 -*-


import glob
import argparse
import os
import time
import sys
import tensorflow as tf
from itertools import count

from config import cfg
import cv2
from model import RPN3D
from utils import *
from utils.fluid_loader import iterate_data

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,
                    help='set batch size') # TOOO change batch_size here
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.0,
                    help='set alpha in los function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=10.0,
                    help='set beta in los function')
parser.add_argument('--output-path', type=str, nargs='?',
                    default='./predictions', help='results output dir')
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                    help='set the flag to True if dumping visualizations')
args = parser.parse_args()

dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(cfg.DATA_DIR, 'train')  # ccx need to change
# val_dir = os.path.join(cfg.DATA_DIR, 'validation')  # ccx need to change
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)


def main(_):
    # TODO: split file support
    with tf.Graph().as_default():
        global save_model_dir
        start_epoch = 0
        global_counter = 0

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                    visible_device_list=cfg.GPU_AVAILABLE,
                                    allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )
        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                learning_rate=args.lr,
                max_gradient_norm=5.0,
                alpha=args.alpha,
                beta=args.beta,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )

            # param init/restore
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(
                    sess, tf.train.latest_checkpoint(save_model_dir))

                start_epoch = model.epoch.eval() + 1
                global_counter = model.global_step.eval() + 1
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            # train and validate
            is_summary, is_summary_image, is_validate = False, False, False

            summary_interval = 5
            summary_val_interval = 10
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            for idx in range(len(model.avail_gpus)):
                count = 0
                concat_feature = []
                with tf.variable_scope("concatfeature", reuse=True):
                    for agent in range(args.single_batch_size):
                        # num = self.vox_k_dynamic[idx][agent]
                        num = model.vox_k_dynamic[idx][agent]
                        concat_feature.append(tf.concat([model.vox_feature[idx][count: count+num],
                                                        model.vox_feature[idx][count: count+num, :, :3] - model.vox_centroid[idx][agent][:3],
                                                        model.vox_feature[idx][count: count+num, :, 3:6] - model.vox_centroid[idx][agent][3:]],axis=2))
                        count += num

                    # self.outputs[idx].set_shape([self.single_batch_size, cfg.INPUT_WIDTH,
                    #                              cfg.INPUT_HEIGHT, cfg.INPUT_DEPTH, 128])
                    # tf.import_graph_def(session.graph_def, input_map={"gpu_0/ScatterNd:0": self.outputs[idx]})
                    concat_feature = tf.concat(concat_feature, axis=0)

                    print(concat_feature)
                model.concat_feature.append(concat_feature)
            # training
            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                batch_time = time.time()
                # TODO ccx get batch data(read data from file)
                for batch in iterate_data(train_dir, sample_rate=0.001, shuffle=True, aug=True, is_testset=False,
                                          batch_size=args.single_batch_size * cfg.GPU_USE_COUNT,
                                          multi_gpu_sum=cfg.GPU_USE_COUNT):

                    counter += 1
                    global_counter += 1

                    if counter % summary_interval == 0:
                        is_summary = True
                    else:
                        is_summary = False

                    start_time = time.time()
                    ret = model.train_step(sess, batch, train=True, summary=is_summary)
                    forward_time = time.time() - start_time
                    batch_time = time.time() - batch_time


                    print(
                        'train: {} @ epoch:{}/{} loss: {:.4f}  forward time: {:.4f} batch time: {:.4f}'.format(
                            counter, epoch, args.max_epoch, ret[0], forward_time,
                            batch_time))
                    with open('log/train.txt', 'a') as f:
                        f.write(
                            'train: {} @ epoch:{}/{} loss: {:.4f} forward time: {:.4f} batch time: {:.4f} \n'.format(
                                counter, epoch, args.max_epoch, ret[0], forward_time,
                                batch_time))

                    # print(counter, summary_interval, counter % summary_interval)
                    if counter % summary_interval == 0:
                        print("summary_interval now")
                        summary_writer.add_summary(ret[-1], global_counter)
                    if counter % 10 == 0:
                        model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'),
                                         global_step=model.global_step)
                    batch_time = time.time()

                sess.run(model.epoch_add_op)

                model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)

                # TODO sample_test_data make a val_dir centroid corespond to batch_size

            print('train done. total epoch:{} iter:{}'.format(
                epoch, model.global_step.eval()))

            # finallly save model
            model.saver.save(sess, os.path.join(
                save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)

