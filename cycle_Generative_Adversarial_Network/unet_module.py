import numpy as np
import tensorflow as tf

def conv_transpose_layer(prev_layer, filter, kernel_size, strides, is_training, alpha):
    x = tf.layers.conv2d_transpose(prev_layer, filter, kernel_size, strides, 'same')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.maximum(x, alpha*x)
    return x

def encoder(input_layer, input_channel, output_channel, max_pooling, name):
    with tf.variable_scope(name):
        if not max_pooling:
            W1 = tf.Variable(tf.random_normal([3, 3, input_channel, output_channel], stddev=0.01))
            L1 = tf.nn.conv2d(input_layer, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.layers.batch_normalization(L1, training=True, trainable=True)
            L1 = tf.nn.relu(L1)

            W2 = tf.Variable(tf.random_normal([3, 3, output_channel, output_channel], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.layers.batch_normalization(L2, training=True, trainable=True)
            L2 = tf.nn.relu(L2)
            return L2

        else:
            L3 = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            W4 = tf.Variable(tf.random_normal([3, 3, input_channel, output_channel], stddev=0.01))
            L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
            L4 = tf.layers.batch_normalization(L4, training=True, trainable=True)
            L4 = tf.nn.relu(L4)

            W5 = tf.Variable(tf.random_normal([3, 3, output_channel, output_channel], stddev=0.01))
            L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
            L5 = tf.layers.batch_normalization(L5, training=True, trainable=True)
            L5 = tf.nn.relu(L5)
        return L5

def decoder(concat_layer, upsample_layer, concat_channel, upsample_channel, output_channel, name):
    with tf.variable_scope(name):
        L13 = conv_transpose_layer(upsample_layer, upsample_channel, 3, 2, True, 0.02)
        L14 = tf.concat([concat_layer, L13], axis=3)

        W15 = tf.Variable(tf.random_normal([3, 3, concat_channel+upsample_channel, concat_channel+upsample_channel], stddev=0.01))
        L15 = tf.nn.conv2d(L14, W15, strides=[1, 1, 1, 1], padding='SAME')
        L15 = tf.layers.batch_normalization(L15, training=True, trainable=True)
        L15 = tf.nn.relu(L15)

        W16 = tf.Variable(tf.random_normal([3, 3, concat_channel+upsample_channel, concat_channel+upsample_channel], stddev=0.01))
        L16 = tf.nn.conv2d(L15, W16, strides=[1, 1, 1, 1], padding='SAME')
        L16 = tf.layers.batch_normalization(L16, training=True, trainable=True)
        L16 = tf.nn.relu(L16)

        W17 = tf.Variable(tf.random_normal([3, 3, concat_channel+upsample_channel, output_channel], stddev=0.01))
        L17 = tf.nn.conv2d(L16, W17, strides=[1, 1, 1, 1], padding='SAME')
        L17 = tf.layers.batch_normalization(L17, training=True, trainable=True)
        L17 = tf.nn.relu(L17)

    return L17