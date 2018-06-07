import os
import time

import tensorflow as tf
from utils.fluid_loader import iterate_single_frame, get_all_frames
from utils.model_utils import load_graph_with_input_map


class Predict(object):

    def __init__(self, frozen_model_filename, batch_size):
        self.batch_size = batch_size
        self.graph, self.acceleration = load_graph_with_input_map(frozen_model_filename, batch_size=self.batch_size)
        """
        node name
        """
        'gpu_0/feature_0'
        'gpu_0/k_dynamics'
        'gpu_0/centroid'
        'gpu_0/coordinate'
        self.screen_size = self.graph.get_tensor_by_name('gpu_0/screen_size:0')
        self.feature = self.graph.get_tensor_by_name('gpu_0/feature:0')
        self.part_feature = self.graph.get_tensor_by_name('gpu_0/part_feature:0')
        self.concat_feature = self.graph.get_tensor_by_name('concat_129:0')
        # accelaration = graph.get_tensor_by_name("gpu_0/MiddleAndRPN_/output/BiasAdd:0")
        self.centroid = self.graph.get_tensor_by_name("gpu_0/centroid:0")
        self.k_dynamics = self.graph.get_tensor_by_name("gpu_0/k_dynamics:0")
        self.coordinate = self.graph.get_tensor_by_name("gpu_0/coordinate:0")
        self.phase = self.graph.get_tensor_by_name("phase:0")
        self.scatter_nd = self.graph.get_tensor_by_name('gpu_0/ScatterNd:0')
        self.voxelwise = self.graph.get_tensor_by_name("gpu_0/Max_1:0")
        #
        self.phase_1 = self.graph.get_tensor_by_name("phase_1:0")

    def caculate_concat_feature(self):
        pass

    def fluidnet_predict(self, batch_size=1, singel_batch=None):

        if singel_batch is None:
            print("single_batch can't be None")
            return
        # predict
        with tf.Session(graph=self.graph) as sess:
            input_dict = dict()
            input_dict[self.phase] = False
            input_dict[self.phase_1] = False
            screen_size_eval = self.screen_size.eval(session=sess, feed_dict={self.screen_size: batch_size})
            concat_feature_all = []
            count = 0
            # print("screen_size_eval:\n", screen_size_eval)
            for screen in range(screen_size_eval):
                k = self.k_dynamics[screen]
                concat_feature_all.append(
                    tf.concat([self.part_feature[count: count + k], self.part_feature[count: count + k, :, :3] - self.centroid[screen]],
                              axis=2))

                count += k
            concat_feature_all = tf.concat(concat_feature_all, axis=0)

            input_dict[self.part_feature] = singel_batch[1][0]
            input_dict[self.k_dynamics] = singel_batch[5][0]
            input_dict[self.centroid] = singel_batch[4][0]
            concat_feature_eval = sess.run(concat_feature_all, input_dict)
            input_dict[self.feature] = concat_feature_eval
            input_dict[self.coordinate] = singel_batch[3][0]
            input_dict[self.phase] = False
            #     input_dict[coordinate] = singel_batch[0]
            accelaration_eval = sess.run(self.acceleration[0], feed_dict=input_dict)

            return accelaration_eval


if __name__ == "__main__":
    test_predict = True
    if test_predict:

        batch_size = 25
        # iter frame data
        data_dir = '/data/datasets/simulation_data'
        train_dir = os.path.join(data_dir, 'water')
        TRAIN_FILES = get_all_frames(train_dir)
        file_path = list(TRAIN_FILES)[5]

        model_dir = "/data/deeplearning/FluidAiNet/save_model/default"
        frozen_model_filename = os.path.join(model_dir, "../frozen_model/frozen_model.pb")
        predict = Predict(frozen_model_filename, batch_size=batch_size)

        start = time.clock()
        result = []
        for batch in iterate_single_frame(train_dir, file_path, batch_size=batch_size):
            accelaration_eval = predict.fluidnet_predict(batch_size=batch_size, singel_batch=batch)
            result.extend(accelaration_eval)
            # if len(result) % 500 == 0:
            print(len(result), 'duration:\n', time.clock() - start)
        print('duration:\n', time.clock() - start)
