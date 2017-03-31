"""
implementation of LeNet as described on:

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

replaced the subsampling layes with max pooling and used Xavier initialization
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten


def conv2d(x, kernel_sz, depth, strides=1):
    weights = tf.get_variable('weights',
                              shape=[kernel_sz, kernel_sz, x.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable('biases', shape=[depth],
                             initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='VALID')
    return tf.nn.bias_add(conv, biases)


def fc(x, out_sz):
    weights = tf.get_variable('weights',
                              shape=[x.get_shape()[1], out_sz],
                              initializer=tf.contrib.layers.xavier_initializer())

    biases = tf.get_variable('biases', shape=[out_sz],
                             initializer=tf.constant_initializer(0.0))

    return tf.add(tf.matmul(x, weights), biases)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')


def Get_LeNet(x, n_classes=10):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    with tf.variable_scope('conv1'):
        conv1 = conv2d(x, kernel_sz=5, depth=6)
        conv1 = tf.nn.relu(conv1)
        pool1 = maxpool2d(conv1)

    # Layer 2: Convolutional. Output = 10x10x16.
    with tf.variable_scope('conv2'):
        conv2 = conv2d(pool1, kernel_sz=5, depth=16)
        conv2 = tf.nn.relu(conv2)
        pool2 = maxpool2d(conv2)

    # Flatten. Input = 5x5x16. Output = 400.
    fc_input = flatten(pool2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    with tf.variable_scope('fc3'):
        fc3 = fc(fc_input, 120)
        fc3 = tf.nn.relu(fc3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    with tf.variable_scope('fc4'):
        fc4 = fc(fc_input, 84)
        fc4 = tf.nn.relu(fc4)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    with tf.variable_scope('out'):
        logits = fc(fc_input, n_classes)

    return logits
