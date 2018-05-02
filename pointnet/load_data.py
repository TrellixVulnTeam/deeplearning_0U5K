
import argparse
import time
import operator
import os
import sys

import h5py
from nose.tools import assert_equal
import numpy as np
import pandas as pd

BASE_DIR = '/data/datasets/simulation_data'

DATA_DIR = os.path.join(BASE_DIR, 'integral/18')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def get_data_files():
    filenames = []
    allfiles = os.listdir(DATA_DIR)
    for filename in allfiles:
        if filename.endswith('.csv'): 
            filenames.append(filename)
    return map(lambda x: os.path.join(DATA_DIR, x), filenames)

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
    return laod_csv(filename)

def load_data_label(filename):
    particles = laod_csv(filename)
    cols = particles.columns
    data_cols = operator.add(list(cols[0:6]), list(cols[7:9]))
    label_cols = cols[15:18]
    data = particles[data_cols]
    label = particles[label_cols]
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

# Shuffle train files
def concat_data_label(train_files, max_points, dimention_data, dimention_label):
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
    print "runtime: %s" % str(running)
    return current_data, current_label

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data