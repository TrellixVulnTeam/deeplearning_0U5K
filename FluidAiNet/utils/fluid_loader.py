
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

import time
import operator
import os

from nose.tools import assert_equal
import numpy as np
import pandas as pd

BASE_DIR = '/data/datasets/simulation_data'
DATA_DIR = os.path.join(BASE_DIR, 'water/0')

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def get_data_files():
    filenames = []
    # TODO multi folders for different frames
    allfiles = os.listdir(DATA_DIR)
    for filename in allfiles:
        if filename.endswith('.csv'):
            filenames.append(filename)
    return map(lambda x: os.path.join(DATA_DIR, x), filenames)

def create_train_files(max_num):
    TRAIN_FILES = []
    files_map = get_data_files()
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
        #if col == 'isFluidSolid':
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

def load_data_label(filename):
    particles = load_data_file(filename)
    cols = particles.columns
    data_cols = operator.add(list(cols[0:6]), list(cols[7:9]))
    label_cols = cols[15:18]
    data = particles[data_cols].values
    label = particles[label_cols].values
    return data, label

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
        current_data_single, current_label_single = load_data_label(TRAIN_FILES[train_file_idxs[fn]])
        current_data[fn] = current_data_single.values[:MAX_POINTS, :]
        current_label[fn]= current_label_single.values[:MAX_POINTS, :]
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
        current_data_single, current_label_single = load_data_label(TRAIN_FILES[train_file_idxs[fn]])
        current_data.append(current_data_single)
        current_label.append(current_label_single)
    running = time.clock() - start
    print("runtime: %s" % str(running))
    return current_data, current_label

if __name__ == '__main__':
    BATCH_SIZE = 2
    TRAIN_FILES = create_train_files(2)
    print(TRAIN_FILES)