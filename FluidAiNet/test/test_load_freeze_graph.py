import os

import numpy as np
import tensorflow as tf
from FluidAiNet.utils.model_utils import *



def sample_Z(m, n, k):
    return np.random.uniform(-8., 8., size=[m, n, k])

if __name__ == '__main__':
    model_dir = "/data/deeplearning/FluidAiNet/save_model/default"
    frozen_model_filename = os.path.join(model_dir, "../frozen_model/frozen_model.pb")
    graph = load_graph(frozen_model_filename)
    """
    [[-3.2216332  -5.21702671 -4.43089962]
     [-3.52895737 -4.99064016 -4.25014257]]
    """
    feature_ = sample_Z(7, 64, 8)
    k_dynamics_ = np.array([0, 7])
    coordinate_ = np.array([[1, 1, 2, 3],
                            [1, 1, 2, 3],
                            [1, 1, 2, 3],
                            [1, 1, 2, 3],
                            [1, 1, 2, 3],
                            [1, 1, 2, 3],
                            [1, 1, 2, 3], ])
    centroid_ = np.array([[0.4, 7.4, 0.4], [0.4, 0.4, 0.4]])
    accelaration = graph.get_tensor_by_name("gpu_0/MiddleAndRPN_/output/BiasAdd:0")
    feature = graph.get_tensor_by_name("gpu_0/feature:0")
    centroid = graph.get_tensor_by_name("gpu_0/centroid:0")
    k_dynamics = graph.get_tensor_by_name("gpu_0/k_dynamics:0")
    coordinate = graph.get_tensor_by_name("gpu_0/coordinate:0")
    phase = graph.get_tensor_by_name("phase:0")
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        accelaration_elv = sess.run(accelaration,
                                    feed_dict={feature: feature_, k_dynamics: k_dynamics_,
                                               coordinate: coordinate_, centroid: centroid_,
                                               phase: False})
    print(accelaration_elv)