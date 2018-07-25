import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tensorflow as tf

from config import cfg
from utils.fluid_loader import iterate_single_frame, get_all_frames
from utils.model_utils import load_graph_with_input_map, load_graph
from utils.fluid_loader import load_data_label

dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(cfg.DATA_DIR, 'train')
avail_gpus = cfg.GPU_AVAILABLE.split(',')
info_project = cfg.INFO_DIR_PROJECT


class Predict(object):

    def __init__(self, frozen_model_filename, batch_size):
        self.batch_size = batch_size
        self.graph, self.acceleration = load_graph_with_input_map(frozen_model_filename, batch_size=self.batch_size)
        # self.graph = load_graph(frozen_model_filename)
        # self.acceleration = self.graph.get_tensor_by_name("gpu_0/MiddleAndReg_/output/BiasAdd:0")
        """
        node name
        """
        'gpu_0/feature_0'
        'gpu_0/k_dynamics'
        'gpu_0/centroid'
        'gpu_0/coordinate'
        self.concat_feature = None
        self.screen_size = self.graph.get_tensor_by_name('gpu_0/screen_size:0')
        self.feature = self.graph.get_tensor_by_name('gpu_0/feature:0')
        self.part_feature = self.graph.get_tensor_by_name('gpu_0/part_feature:0')

        # accelaration = graph.get_tensor_by_name("gpu_0/MiddleAndRPN_/output/BiasAdd:0")
        self.centroid = self.graph.get_tensor_by_name("gpu_0/centroid:0")
        self.k_dynamics = self.graph.get_tensor_by_name("gpu_0/k_dynamics:0")
        self.coordinate = self.graph.get_tensor_by_name("gpu_0/coordinate:0")
        self.phase = self.graph.get_tensor_by_name("phase:0")
        # self.scatter_nd = self.graph.get_tensor_by_name('gpu_0/ScatterNd:0')
        self.scatter_nd = self.graph.get_tensor_by_name('gpu_0/ScatterNd_1:0')  # reconstruct by batch_size
        # self.voxelwise = self.graph.get_tensor_by_name("gpu_0/Max_1:0")
        #
        self.phase_1 = self.graph.get_tensor_by_name("phase_1:0")

        self.caculate_concat_feature(self.batch_size)

    def caculate_concat_feature(self, batch_size):
        for idx, dev in enumerate(avail_gpus):
            with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
                with tf.variable_scope("concatfeature", reuse=True):
                    concat_feature_all = []
                    count = 0
                    # print("screen_size_eval:\n", screen_size_eval)
                    for screen in range(batch_size):
                        k = self.k_dynamics[screen]
                        concat_feature_all.append(
                            tf.concat([self.part_feature[count: count + k],
                                       self.part_feature[count: count + k, :, :3] - self.centroid[screen][:3],
                                       self.part_feature[count: count + k, :, 3:6] - self.centroid[screen][3:6]],
                                      axis=2))
                        count += k
                    self.concat_feature = tf.concat(concat_feature_all, axis=0)
                    print(self.concat_feature)

    def fluidnet_predict(self, batch_size=1, singel_batch=None):

        if singel_batch is None:
            print("single_batch can't be None")
            return
        # predict
        with tf.Session(graph=self.graph) as sess:
            if batch_size != self.batch_size:
                print('')
                self.caculate_concat_feature(batch_size)

            input_dict = dict()
            input_dict[self.phase] = False
            input_dict[self.phase_1] = False
            # screen_size_eval = self.screen_size.eval(session=sess, feed_dict={self.screen_size: batch_size})
            input_dict[self.part_feature] = singel_batch[1][0]
            input_dict[self.k_dynamics] = singel_batch[5][0]
            input_dict[self.centroid] = singel_batch[4][0]
            start = time.clock()
            concat_feature_eval = sess.run(self.concat_feature, input_dict)
            print('concat feature:\n', time.clock()-start)
            input_dict[self.feature] = concat_feature_eval
            input_dict[self.coordinate] = singel_batch[3][0]
            start = time.clock()
            accelaration_eval = sess.run(self.acceleration[0], feed_dict=input_dict)
            print('accelaration_eval:\n', time.clock() - start)
            return accelaration_eval


def plot_3d_scater_df(particles_df, is_solid, figure_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cols = particles_df.columns
    particles_df = particles_df[particles_df[cols[6]] == is_solid]
    x = particles_df[cols[0]]
    y = particles_df[cols[1]]
    z = particles_df[cols[2]]
    ax.scatter(x, y, z, c="b", marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(figure_path)
    plt.show()


def plot_file(file_path):
    data, label, index = load_data_label(file_path, isvalues=False)
    plot_3d_scater_df(data, 0)


def write_acc_2_file(f, accelaration_eval):
    for acc in accelaration_eval:
        for num in acc:
            f.write(str(num))
            f.write(',')
        f.write('\n')


def iter_batch_size(frozen_model_filename, init_batch_size, f, predict, result, start, train_dir, file_path,
                    data_new=None, index_new=None, sample_rate=1, batch_size=25):
    for batch, batch_size in iterate_single_frame(train_dir, file_path, data_new=data_new, index_new=index_new,
                                                  sample_rate=sample_rate, batch_size=batch_size):
        if batch_size != init_batch_size:
            predict = Predict(frozen_model_filename, batch_size=batch_size)
        accelaration_eval = predict.fluidnet_predict(batch_size=batch_size, singel_batch=batch)
        print('**********', accelaration_eval)
        write_acc_2_file(f, accelaration_eval)
        result.extend(accelaration_eval)
        if len(result) % 500 == 0:
            print(len(result), 'duration:\n', time.clock() - start)
    print('duration:\n', time.clock() - start)
    # return accelaration_eval


def get_local_time():
    return time.asctime(time.localtime(time.time()))


def forward():
    test_predict = True
    zero_time = get_local_time()
    print('start: ', zero_time)
    if test_predict:

        init_batch_size = 25
        forward_step = 3
        # load trained model
        model_dir = "/data/info/FluidAiNet/save_model/default"
        frozen_model_filename = os.path.join(model_dir, "../frozen_model/frozen_model.pb")
        predict = Predict(frozen_model_filename, batch_size=init_batch_size)

        # create dir for current task
        # os.makedirs(path) multifolders os.mkdir(path) singlefolder
        localtime = time.asctime(time.localtime(time.time()))
        unique_dir = '_'.join(str(localtime).split(' '))

        forward_dir = os.path.join(info_project, 'predict', unique_dir)
        if not os.path.exists(forward_dir):
            os.mkdir(forward_dir)
        acc_dir = os.path.join(forward_dir, 'acc')
        if not os.path.exists(acc_dir):
            os.mkdir(acc_dir)
        figure_dir = os.path.join(forward_dir, 'figure')
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)

        # load init data
        TRAIN_FILES = get_all_frames(train_dir)
        file_path = list(TRAIN_FILES)[0]
        data, label, index = load_data_label(file_path, isvalues=False)
        figure_path = os.path.join(figure_dir, 'figure_0.png')
        plot_3d_scater_df(data[index[0]:index[-1]], 0, figure_path)

        delta_t = 0.000608

        start = time.clock()

        result = []

        for step in range(forward_step):
            f = open(os.path.join(acc_dir, 'accelaration_%d.txt' % step), 'w+')
            if step == 0:
                iter_batch_size(frozen_model_filename, init_batch_size, f, predict, result, start,
                                                    train_dir, file_path, data_new=None, index_new=None, batch_size=25)
            else:
                # step after the first step and update with the previous data
                iter_batch_size(frozen_model_filename, init_batch_size, f, predict, result, start,
                                                    train_dir, file_path, data_new=data.copy(), index_new=index, batch_size=25)
            f.close()
            # todo update velocity and accrlaration
            # updata velocity
            try:
                df_acc = pd.DataFrame(result)
                acc_num = len(df_acc.index)
                cols = data.columns
                for i in range(3):
                    data[cols[i + 3]][0: acc_num] = data[cols[i + 3]][0: acc_num] + \
                                                    df_acc[df_acc.columns[i]] * delta_t
                # update position
                for i in range(3):
                    data[cols[i]][0: acc_num] = data[cols[i]][0: acc_num] + \
                                                data[cols[i + 3]][0: acc_num] * delta_t
                figure_path = os.path.join(figure_dir, 'figure_%d.png' % (step+1))
                plot_3d_scater_df(data[index[0]:index[-1]], 0, figure_path)
            except Exception as e:
                print('update velocity and acc error')
                print(e)
    print('start on ', zero_time, ' and end on ', get_local_time(), '\n')


if __name__ == "__main__":
    import os
    forward()
