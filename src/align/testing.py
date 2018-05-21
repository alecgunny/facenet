from align.detect_face import get_graph_and_func
import tensorflow as tf
import time
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)
parser.add_argument('max_batch_size', type=int)
args = parser.parse_args()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
tf_graph_def, tf_func, trt_graph_def, trt_func = get_graph_and_func(gpu_options, None, args.net, args.max_batch_size)

input_shapes = {'pnet': (150, 150, 3), 'rnet': (24, 24, 3), 'onet': (48, 48, 3)}
inp = np.random.randn(args.max_batch_size, *input_shapes[args.net]).astype('float32')

def time_func(func, inp, iters=100):
    start = time.time()
    for i in range(iters):
        func(inp)
    return time.time() - start

def get_node_by_name(gdef, name):
    for node in gdef.node:
        if node.name=='name':
            return node

tf_warm_up = time_func(tf_func, inp, 20)
trt_warm_up = time_func(trt_func, inp, 20)

tf_time = time_func(tf_func, inp)
trt_time = time_func(trt_func, inp)

print('')
print('TensorFlow Execution Time: {:0.4f} s'.format(tf_time))
print('TensorRT Execution Time: {:0.4f} s'.format(trt_time))
