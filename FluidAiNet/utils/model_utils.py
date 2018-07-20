import os
import time

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import inspect_checkpoint


# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
# tf.train.write_graph(sess.graph_def, 'model/', 'graph.pb', as_text=False)


def freeze_graph(model_folder, output_node_names=None):

    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print('input_checkpoint:\n', input_checkpoint)

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    print('absolute_model_folder:\n', absolute_model_folder)
    output_graph = os.path.join(model_folder, '../frozen_model', 'frozen_model.pb')
    print('output_graph:\n', output_graph)

    # Before exporting our graph, we need to precise what is our output node
    # NOTE: this variables is plural, because you can have multiple output nodes
    if not output_node_names:
        output_node_names = "label,fake_image" # TODO this can force you to figure out what you really want, define your aims.

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    # fix batch norm nodes


    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # for node in input_graph_def.node:
        #     if node.op == 'RefSwitch':
        #         node.op = 'Switch'
        #         for index in range(len(node.input)):
        #             if 'moving_' in node.input[index]:
        #                 node.input[index] = node.input[index] + '/read'
        #     elif node.op == 'AssignSub':
        #         node.op = 'Sub'
        #         if 'use_locking' in node.attr: del node.attr['use_locking']
        # We use a built-in TF helper to export variables to constant
        all_nodes("mean")
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, 
            input_graph_def, 
            output_node_names.split(",")  # We split on comma for convenience
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename, prefix=''):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name=prefix)
    return graph


def load_graph_with_input_map(frozen_graph_filename, prefix='', batch_size=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:

        tf.import_graph_def(graph_def, name=prefix)
        coordinate = graph.get_tensor_by_name("gpu_0/coordinate:0")
        voxelwise = graph.get_tensor_by_name("gpu_0/Mean_2:0")
        scatter_nd = graph.get_tensor_by_name("gpu_0/ScatterNd:0")
        # holder = tf.placeholder(tf.float32,shape=(3, 40, 40, 50, 1))
        new_scatter_nd = tf.scatter_nd(coordinate, voxelwise, shape=[batch_size, 40, 40, 50, 128])
        print(new_scatter_nd)
        # conv3D = graph.get_tensor_by_name('gpu_0/MiddleAndRPN_/conv1/Conv3D:0')
        # print(conv3D)
        # conv3D.input = new_scatter_nd
        # print(conv3D.input.shape)
        # graph.graph_place()
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
    # with tf.Graph().as_default() as graph:
        # scatter_nd = graph.get_tensor_by_name('gpu_0/ScatterNd:0')
        # saver = tf.train.import_meta_graph(graph_def, name=prefix, input_map={'gpu_0/ScatterNd': holder})
        # z = tf.import_graph_def(graph_def, name=prefix, input_map={'gpu_0/ScatterNd:0': new_scatter_nd},return_elements=["gpu_0/MiddleAndRPN_/conv1/Conv3D:0"])
        accerlation = tf.import_graph_def(graph_def, name=prefix, input_map={'gpu_0/ScatterNd:0': new_scatter_nd}, return_elements=["gpu_0/MiddleAndReg_/output/BiasAdd:0"])
        print(accerlation)
        # _ = tf.train.import_meta_graph(graph_def, name=prefix, input_map={'gpu_0/ScatterNd': new_scatter_nd})
    return graph, accerlation


def all_nodes(node_key=""):
    return [n.name for n in tf.get_default_graph().as_graph_def().node if n.name.rfind(node_key) > -1]


class LoadGraphTest:
    def __init__(self, model_dir, frozen_model_filename):
        self.model_dir = model_dir
        self.frozen_model_filename = frozen_model_filename
    
    def test_freezen_model(self, output_node_names=None):
        freeze_graph(model_dir, output_node_names)

    def test_load_graph(self):
        pass


if __name__ == '__main__':
    print("sync")
    model_dir = "/data/info/FluidAiNet/save_model/default"
    # freeze_graph(model_dir, output_node_names="gpu_0/screen_size,gpu_0_1/concatfeature/concat_1,gpu_0/MiddleAndReg_/output/BiasAdd")
    # frozen_model_filename = os.path.join(model_dir, "../frozen_model/frozen_model.pb")
    # graph, _ = load_graph_with_input_map(frozen_model_filename, batch_size=3)
    # # graph = load_graph(frozen_model_filename)
    # # scatter_nd = graph.get_tensor_by_name('gpu_0/ScatterNd:0')
    # # print(scatter_nd)
    # with tf.Session(graph=graph) as sess:
    #     #     sess.run(tf.global_variables_initializer())
    #     print(all_nodes("Mean_2"))
    frozen_model_filename = os.path.join(model_dir, "../frozen_model/frozen_model.pb")
    # load_graph_with_input_map(frozen_model_filename, batch_size=25)

