import os
import multiprocessing
import numpy as np
from nose.tools import assert_equal

from config import cfg

data_dir = 'velodyne'


"""
screen size (-8, -8, -8) (8, 8, 12)
spacing_r 0.1f # diameter 
amoothRadius_h  2*spacing_r # so i decide the voxel_size 4*spacing_r
particle size - radius
x,y,z(21,21,26) (-1, -1, -1) (1, 1, 1.5)
theoreticlly, 
"""

def fluid_process_pointcloud(point_cloud, fluid_identification,cls=cfg.DETECT_OBJ):
    centroid = point_cloud[fluid_identification][:3]
    print(centroid)
    if cls == 'Fluid':
        print('Fluid')
        scene_size = np.array([16, 16, 20], dtype=np.float32)
        voxel_size = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        grid_size = np.array([40, 40, 50], dtype=np.int64)
        lidar_coord = np.array([8, 8, 8], dtype=np.float32)
        max_point_number = 64
        # return
    # FIXME  ccx AWESOME
    print(point_cloud)
    shifted_coord = point_cloud[:, :3] + lidar_coord
    voxel_index = np.floor(
        shifted_coord[:, :] / voxel_size).astype(np.int)  # int lower than num

    print(voxel_index.shape)
    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)  # K record the number of voxels
    T = max_point_number
    print(K)
    assert_equal(T, 64)
    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 11), dtype=np.float32) # position, velocity, isFluid, index, relative position

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        # mark the voxel,the position tuple of voxel as list(index_buffer)'s key
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :8] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - centroid


    voxel_dict = {'feature_buffer': feature_buffer,  # (K, T, 7)
                  'coordinate_buffer': coordinate_buffer,  # (K, 3)
                  'number_buffer': number_buffer}  # (K,)
    return voxel_dict

def process_pointcloud(point_cloud, cls=cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    elif cls == 'Fluid':
        print('Fluid')
        scene_size = np.array([16, 16, 20], dtype=np.float32)
        voxel_size = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([8, 8, 8], dtype=np.float32)
        max_point_number = 64
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)# fix coordinate to positive(z,y,x),which is the max num of minus
        max_point_number = 45

        np.random.shuffle(point_cloud)
    # FIXME  ccx AWESOME
    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int) # int lower than num

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer) # K record the number of voxels
    T = max_point_number # TODO ccx set T reasonable?

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        # mark the voxel,the position tuple of voxel as list(index_buffer)'s key
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer, # (K, T, 7)
                  'coordinate_buffer': coordinate_buffer, # (K, 3)
                  'number_buffer': number_buffer} # (K,)
    return voxel_dict

if __name__ == "__main__":
#     import fluid_loader

#     BATCH_SIZE = 2
#     TRAIN_FILES = fluid_loader.create_train_files(BATCH_SIZE)
#     print(TRAIN_FILES)
#     pointcloud, _, _ = fluid_loader.concat_data_label_all(TRAIN_FILES, 8, 3)
#     voxel_index = fluid_process_pointcloud(pointcloud[0],1)
    pass