#!/usr/bin/env python
# encoding: utf-8

"""
@author: changxin
@software: PyCharm
@file: fluid_loader.py
@time: 2018/5/4 15:04
"""
from __future__ import print_function
from __future__ import division

import math
import multiprocessing
import operator
import os
import time

from nose.tools import assert_equal
import numpy as np
import pandas as pd

from utils.preprocess import fluid_process_pointcloud
# from preprocess import fluid_process_pointcloud

BASE_DIR = '/data/datasets/simulation_data'
DATA_DIR = os.path.join(BASE_DIR, 'water')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


class Processor(object):
    def __init__(self, particles, labels, index, data_dir, aug, is_testset):
        self.particles = particles
        self.labels = labels
        self.index = index
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset
        self.common = None
        self.voxel = fluid_process_pointcloud(self.particles)

    def __call__(self, load_index):
        label = self.labels[load_index]
        centroid = fluid_process_pointcloud(self.particles, fluid_identification=load_index)
        self.voxel['centroid'] = centroid

        ret = [self.voxel, label]

        return ret


def get_all_frames(data_dir=DATA_DIR):
    dirs = os.listdir(data_dir)
    frames = []
    for item in dirs:
        screen_path = os.path.join(data_dir, item)
        allfiles = os.listdir(screen_path)
        # for i in range(10):
        #     np.random.shuffle(allfiles)
        #     print(allfiles)
        np.random.shuffle(allfiles)
        frames.extend(map(lambda x: os.path.join(item, x), allfiles))
    return map(lambda x: os.path.join(data_dir, x), frames)


# for test return subset of frames
def create_train_files(max_num):
    TRAIN_FILES = []
    files_map = get_all_frames()
    i = 0
    max_count = 2
    for item in files_map:
        if i == max_count:
            break
        TRAIN_FILES.append(item)
        i = i + 1

    assert_equal(len(TRAIN_FILES), max_count)
    return TRAIN_FILES


def convert_str_float(frame_particles):
    fps = pd.DataFrame(frame_particles[1:], columns=frame_particles[0])
    fps = fps[fps.columns[:-1]]
    for col in fps.columns:
        # if col == 'isFluidSolid':
        fps[col] = fps[col].astype(float)
    return fps


def laod_csv(filename):
    frame_particles = np.loadtxt(
            filename, dtype=np.str, delimiter=",")
    return convert_str_float(frame_particles)


def load_data_file(filename):
    suffix = filename.split('/')[-1].split('.')[-1]
    print(suffix)
    if suffix == 'csv':
        return laod_csv(filename)


def load_data_label(filename, isvalues=True):
    particles = load_data_file(filename)
    try:
        cols = particles.columns
    except Exception as e:
        print(e)
        print('csv file maybe contain no data :{} '.format(filename))
        raise IOError('file none')
    data_cols = operator.add(list(cols[0:6]), list(cols[7:9]))  # extrat timestep
    label_cols = cols[15:18]

    isfluid = cols[7]
    fluid_parts = particles[particles[isfluid] == 0]
    index = fluid_parts.index
    if isvalues:
        # only fluid parts
        # data = particles[data_cols].values
        data = fluid_parts[data_cols].values
        label = fluid_parts[label_cols].values
    else:
        # data = particles[data_cols]
        data = fluid_parts[data_cols]
        label = fluid_parts[label_cols]

    return data, label, index


def shuffle_label(labels):
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    return idx


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,N... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], idx


def concat_data_label(train_files, max_points, dimention_data, dimention_label):
    """
    intercept max_points
    """
    TRAIN_FILES = train_files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    def get_array(shape):
        return np.empty(shape=shape)
    FRAMES_NUM = len(TRAIN_FILES)
    MAX_POINTS = max_points
    DIMENTION_DATA = dimention_data
    DIMENTION_LABEL = dimention_label
    data_shape = (FRAMES_NUM, MAX_POINTS, DIMENTION_DATA)
    """
    Different from the classification task, our lable is for every particle, we record label with (frame, particle index \
    , three-dimentions accelaration)(BxNx3)
    """
    label_shape = (FRAMES_NUM, MAX_POINTS, DIMENTION_LABEL)
    current_data = get_array(data_shape)
    current_label = get_array(label_shape)
    start = time.clock()
    for fn in range(len(TRAIN_FILES)):
        current_data_single, current_label_single, _ = load_data_label(TRAIN_FILES[train_file_idxs[fn]])
        current_data[fn] = current_data_single.values[:MAX_POINTS, :]
        current_label[fn] = current_label_single.values[:MAX_POINTS, :]
    running = time.clock() - start
    print("runtime: %s" % str(running))
    return current_data, current_label


def concat_data_label_all(train_files, dimention_data, dimention_label):
    """
    use all data in a file
    """
    TRAIN_FILES = train_files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    DIMENTION_DATA = dimention_data
    DIMENTION_LABEL = dimention_label
    """
    Different from the classification task, our lable is for every particle, we record label with (frame, particle index \
    , three-dimentions accelaration)(BxNx3)
    """
    current_data = []
    current_label = []
    start = time.clock()
    for fn in range(len(TRAIN_FILES)):
        current_data_single, current_label_single, _ = load_data_label(TRAIN_FILES[train_file_idxs[fn]])
        current_data.append(current_data_single)
        current_label.append(current_label_single)
    running = time.clock() - start
    print("runtime: %s" % str(running))
    return current_data, current_label


# global pool
TRAIN_POOL = multiprocessing.Pool(5)


def iterate_single_frame(data_dir, file_name, batch_size, data_new=None, index_new=None, sample_rate=1, multi_gpu_sum=1):
    # frame_file_name = os.path.join(data_dir, file_name)
    data, label, index = load_data_label(file_name)
    if data_new is not None and index_new is not None:
        data = data_new
        index = index_new
    nums = len(index)
    indices = list(range(nums))
    num_batches = int(math.floor(nums / float(batch_size)))
    interval = int(1 / sample_rate)

    extra = nums % batch_size
    if extra > 0:
        num_batches += 1

    proc = Processor(data, label, index, data_dir, False, False)
    # only different with centroid
    for batch_idx in range(0, num_batches, interval):
        print(batch_idx, ' of ', num_batches)
        start_idx = batch_idx * batch_size
        if extra > 0 and batch_idx == num_batches - 1:
            batch_size = extra
        excerpt = indices[start_idx:start_idx + batch_size]
        rets = TRAIN_POOL.map(proc, excerpt)

        voxel = [ret[0] for ret in rets]
        assert_equal(len(voxel), batch_size)  # in final step the size may less than init batch_size
        labels = [ret[1] for ret in rets]

        # only for voxel -> [gpu, k_single_batch, ...]
        vox_feature, vox_number, vox_coordinate, vox_centroid, vox_k_dynamic = [], [], [], [], []
        vox_labels = []
        # TODO ccx if bach_size smalls than multi_gpu_sum
        single_batch_size = int(batch_size / multi_gpu_sum)
        for idx in range(multi_gpu_sum):
            label = labels[idx * single_batch_size:(idx + 1) * single_batch_size]
            _, per_vox_feature, per_vox_number, per_vox_coordinate, per_vox_centroid, per_vox_k_dynamic = build_input(
                voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
            # a batch concate all files together ∑K
            vox_labels.append(label)
            vox_feature.append(per_vox_feature)
            vox_number.append(per_vox_number)
            vox_coordinate.append(per_vox_coordinate)
            vox_centroid.append(per_vox_centroid)
            vox_k_dynamic.append(per_vox_k_dynamic)
            # print(vox_k_dynamic)
        ret = (
            np.array(vox_labels),
            np.array(vox_feature),
            np.array(vox_number),
            np.array(vox_coordinate),
            np.array(vox_centroid),
            np.array(vox_k_dynamic)
        )

        yield ret, batch_size


def iterate_data(data_dir, sample_rate=1, shuffle=False, aug=False, is_testset=False, batch_size=1, multi_gpu_sum=1):
    TRAIN_FILES = get_all_frames(data_dir)
    for f in TRAIN_FILES:
        try:
            data, label, index = load_data_label(f)
        except Exception as e:
            print(e)
            continue

        # TODO the common part of feature
        nums = len(index)
        indices = list(range(nums))
        if shuffle:
            np.random.shuffle(indices)
        num_batches = int(math.floor(nums / float(batch_size))) # about 1W/25
        
        interval = int(1/sample_rate)  # num_batches / interval = num_batches * 0.01

        extra = nums % batch_size
        if extra > 0:
            num_batches += 1

        proc = Processor(data, label, index, data_dir, aug, is_testset)
        # only different with centroid
        for batch_idx in range(0, num_batches, interval):
            start_idx = batch_idx * batch_size
            if extra > 0 and batch_idx == num_batches - 1 :
                excerpt = indices[start_idx:start_idx + extra]
            else:
                excerpt = indices[start_idx:start_idx + batch_size]

            # every batch process 'batch_size' particle, but particle feature 1 part a time,concate them together as one batch.
            rets = TRAIN_POOL.map(proc, excerpt)

            voxel = [ret[0] for ret in rets]
            assert_equal(len(voxel), batch_size)
            labels = [ret[1] for ret in rets]

            # only for voxel -> [gpu, k_single_batch, ...]
            vox_feature, vox_number, vox_coordinate, vox_centroid, vox_k_dynamic = [], [], [], [], []
            vox_labels = []
            # TODO ccx if bach_size smalls than multi_gpu_sum
            single_batch_size = int(batch_size / multi_gpu_sum)
            for idx in range(multi_gpu_sum):
                label = labels[idx * single_batch_size:(idx + 1) * single_batch_size]
                _, per_vox_feature, per_vox_number, per_vox_coordinate, per_vox_centroid, per_vox_k_dynamic = build_input(
                    voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
                # a batch concate all files together ∑K
                vox_labels.append(label)
                vox_feature.append(per_vox_feature)
                vox_number.append(per_vox_number)
                vox_coordinate.append(per_vox_coordinate)
                vox_centroid.append(per_vox_centroid)
                vox_k_dynamic.append(per_vox_k_dynamic)
                # print(vox_k_dynamic)
            ret = (
                np.array(vox_labels),
                np.array(vox_feature),
                np.array(vox_number),
                np.array(vox_coordinate),
                np.array(vox_centroid),
                np.array(vox_k_dynamic),
                TRAIN_FILES,
                excerpt
            )

            yield ret


# Fixme iterate all sample test data
def sample_test_data(data_dir, batch_size=1, multi_gpu_sum=1):
    TEST_FILES = list(get_all_frames(data_dir))
    if len(TEST_FILES) < 1:
        raise ValueError("no test files")
    data, label, index = load_data_label(TEST_FILES[0])
    # TODO the common part of feature
    nums = len(index)
    indices = list(range(nums))
    # num_batches = int(math.floor(nums / float(batch_size)))

    proc = Processor(data, label, index, data_dir, False, False)
    # only different with centroid
    # for batch_idx in range(num_batches):

    excerpt = indices[0:batch_size]

    rets = TRAIN_POOL.map(proc, excerpt)

    voxel = [ret[0] for ret in rets]
    assert_equal(len(voxel), batch_size)
    labels = [ret[1] for ret in rets]

    # only for voxel -> [gpu, k_single_batch, ...]
    vox_feature, vox_number, vox_coordinate, vox_centroid, vox_k_dynamic = [], [], [], [], []
    vox_labels = []
    # TODO ccx if bach_size smalls than multi_gpu_sum
    single_batch_size = int(batch_size / multi_gpu_sum)
    for idx in range(multi_gpu_sum):
        label = labels[idx * single_batch_size:(idx + 1) * single_batch_size]
        _, per_vox_feature, per_vox_number, per_vox_coordinate, per_vox_centroid, per_vox_k_dynamic = build_input(
            voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
        # a batch concate all files together ∑K
        vox_labels.append(label)
        vox_feature.append(per_vox_feature)
        vox_number.append(per_vox_number)
        vox_coordinate.append(per_vox_coordinate)
        vox_centroid.append(per_vox_centroid)
        vox_k_dynamic.append(per_vox_k_dynamic)
        # print(vox_k_dynamic)
    ret = (
        np.array(vox_labels),
        np.array(vox_feature),
        np.array(vox_number),
        np.array(vox_coordinate),
        np.array(vox_centroid),
        np.array(vox_k_dynamic)
    )

    return ret


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    centroid_list = []
    k_dinamic_list = []
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        centroid_list.append(voxel_dict['centroid'])
        k_dinamic_list.append(voxel_dict['k_dynamic'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(
            np.pad(coordinate, ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))  # ccx add index for [[i, x1, y1, z1], [i, x2, y2, z2],...,]

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    centroid = np.array(centroid_list)
    k_dynamics = np.array(k_dinamic_list)
    return batch_size, feature, number, coordinate, centroid, k_dynamics


if __name__ == '__main__':
    # BATCH_SIZE = 2
    # TRAIN_FILES = create_train_files(2)
    # print(TRAIN_FILES)
    # data_dir = '/data/datasets/simulation_data'
    # train_dir = os.path.join(data_dir, 'water')
    # batch_size = 1000
    # singel_batch = None
    # for batch in iterate_data(train_dir, batch_size=batch_size):
    #     singel_batch = batch
    #     break
    get_all_frames(data_dir=TRAIN_DATA_DIR)