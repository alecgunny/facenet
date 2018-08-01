from align.detect_face import get_network_graphs_and_funcs
import tensorflow as tf
import time
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('net', type=str)
parser.add_argument('max_batch_size', type=int)
args = parser.parse_args()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
tf_graph_def, tf_func, \
  trt_16_graph_def, trt_16_func, \
  trt_32_graph_def, trt_32_func = get_network_graphs_and_funcs(gpu_options, None, args.net, args.max_batch_size)

input_shapes = {'pnet': (150, 150, 3), 'rnet': (24, 24, 3), 'onet': (48, 48, 3)}
inp = np.random.randn(args.max_batch_size, *input_shapes[args.net]).astype('float32')

def time_func(func, inp, iters=100):
    times = []
    for i in range(iters):
        start = time.time()
        func(inp)
        times.append(time.time() - start)
    return times

def get_node_by_name(gdef, name):
    for node in gdef.node:
        if node.name=='name':
            return node

tf_warm_up = time_func(tf_func, inp, 20)
trt16_warm_up = time_func(trt_16_func, inp, 20)
trt32_warm_up = time_func(trt_32_func, inp, 20)

tf_times = time_func(tf_func, inp)
trt16_times = time_func(trt_16_func, inp)
trt32_times = time_func(trt_32_func, inp)

mean_tf_time = np.mean(tf_times) * 1000
mean_trt16_time = np.mean(trt16_times) * 1000
mean_trt32_time = np.mean(trt32_times) * 1000

print('Benchmarking for 100 iterations at batch size {}'.format(args.max_batch_size))
print('Average TensorFlow Execution Time: {:0.4f} ms'.format(mean_tf_time))
print('Average TensorRT FP16 Execution Time: {:0.4f} ms'.format(mean_trt16_time))
print('Average TensorRT FP32 Execution Time: {:0.4f} ms'.format(mean_trt32_time))
