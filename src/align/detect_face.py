""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
#from math import floor
import cv2
import os

def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path, encoding='latin1').item() #pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def build_trt_graph_def(
            self,
            sess,
            graph,
            output_names,
            max_batch_size,
            precision):
        # for TensorRT, freeze the variables to constants to make a frozen graph
        # then build the TensorRT graph
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_names)
        return trt.create_inference_graph(
            graph_def,
            outputs=output_names,
            max_batch_size=max_batch_size,
            precision_mode=precision,
            minimum_segment_size=2)

    def get_graph_from_graph_def(
            self,
            graph_def,
            input_name,
            output_names):
        g = tf.Graph()
        with g.as_default():
            graph_tensors = tf.import_graph_def(
                graph_def=graph_def,
                return_elements=[input_name]+output_names)
            input = graph_tensors.pop(0).outputs[0]
            outputs = [i.outputs[0] for i in graph_tensors]
        return g, input, outputs

    def build_inference_fcn(
            self,
            sess,
            graph,
            input_name,
            output_names,
            use_trt=False,
            max_batch_size=128,
            precision='fp16'):
        if not use_trt:
            input_name = input_name + ":0"
            output_names = [name+":0" for name in output_names]
            return lambda img: sess.run(output_names, feed_dict={input_name:img})

        trt_graph_def = self.build_trt_graph_def(sess, graph, output_names, max_batch_size, precision)
        trt_graph, input, outputs = self.get_graph_from_graph_def(trt_graph_def, input_name, output_names)

        sess = tf.Session(graph=trt_graph)
        return lambda img: sess.run(outputs, feed_dict={input: img})

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            neg_inp = tf.scalar_mul(-1, inp)
            relu = tf.nn.relu(inp)
            neg_relu = tf.nn.relu(neg_inp)
            output = relu + tf.multiply(alpha, tf.scalar_mul(-1, neg_relu))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc


    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax
    
class PNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))

        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
        
class RNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1,name='prob1'))

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))


def get_network_graphs_and_funcs(gpu_options, model_path, model, max_batch_size):
    if not model_path:
        model_path,_ = os.path.split(os.path.realpath(__file__))

    if model == 'pnet':
        network = PNet
        input_shape = (None, 150, 150, 3)
        output_names = ['conv4-2/BiasAdd', 'prob1']
        weights = 'det1.npy'

    elif model == 'onet':
        network = ONet
        input_shape = (None,48,48,3)
        output_names = ['conv6-2/conv6-2', 'conv6-3/conv6-3', 'prob1']
        weights = 'det3.npy'

    elif model == 'rnet':
        network = RNet
        input_shape = (None,24,24,3)
        output_names = ['conv5-2/conv5-2', 'prob1']
        weights = 'det2.npy'
    else:
        raise ValueErorr('Unknown model {}'.format(model))

    # build the initial model graph
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        sess = tf.Session(config=config)
        with sess.as_default():
            data = tf.placeholder(tf.float32, input_shape, 'input')
            net = network({'data': data})
            net.load(os.path.join(model_path, weights), sess)

    # now convert to graph def
    input_name = data.name.split(":")[0]
    regular_graph_def = graph.as_graph_def()
    regular_func = lambda img: sess.run([name+":0" for name in output_names], feed_dict={data: img})

    # build our half precision graph definition
    trt16_graph_def = net.build_trt_graph_def(
        sess,
        graph,
        output_names,
        max_batch_size=max_batch_size,
        precision='fp16')

    # use this to get an actual graph with it associated outputs
    trt16_graph, input, outputs = net.get_graph_from_graph_def(trt16_graph_def, input_name, output_names)

    # build a session to execute the graph
    trt16_sess = tf.Session(config=config, graph=trt16_graph)

    # finally a wrapper function for calling on images
    outputs = ["import/{}:0".format(name) for name in output_names]
    trt16_func = lambda img: trt16_sess.run(outputs, feed_dict={'import/input:0': img})

    # repeat everything for full precision
    trt32_graph_def = net.build_trt_graph_def(
        sess,
        graph,
        output_names,
        max_batch_size=max_batch_size,
        precision='fp32')
    trt32_graph, input, outputs = net.get_graph_from_graph_def(trt32_graph_def, input_name, output_names)
    trt32_sess = tf.Session(config=config, graph=trt32_graph)
    outputs = ["import/{}:0".format(name) for name in output_names]
    trt32_func = lambda img: trt32_sess.run(outputs, feed_dict={'import/input:0': img})

    return regular_graph_def, regular_func, trt16_graph_def, trt16_func, trt32_graph_def, trt32_func    
