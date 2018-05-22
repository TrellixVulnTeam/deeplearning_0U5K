import os

from utils import fluid_loader as fl
from utils import preprocess

BASE_DIR = '/data/datasets/simulation_data'
DATA_DIR = os.path.join(BASE_DIR, 'water')

class TestFluid(object):
    def __init__(self):
        self.data_dir = DATA_DIR
    def test_iter_data(self):
        for item in fl.iterate_data(self.data_dir):
            print(item)
        
if __name__ == '__main__':
    """
    BATCH_SIZE = 2
    TRAIN_FILES = fl.create_train_files(BATCH_SIZE)
    print(TRAIN_FILES)
    pointcloud, _, _ = fl.load_data_label(TRAIN_FILES[0])
    voxel_index = preprocess.fluid_process_pointcloud(pointcloud, 10021)
    print(voxel_index['feature_buffer'].shape)
    """
    for item in fl.iterate_data(DATA_DIR, batch_size=2):
        print(item[0])
        break

