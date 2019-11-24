import tensorflow as tf
import numpy as np
from tqdm import tqdm

def inference(image_batch, batch_size, num_classes):
    with tf.variable_scope('Conv_1') as scope:
        weights = tf.get_variable('weights_1',
                                    shape = [3,3,1,16],
                                    dtype = tf.float32,
                                    initializer = tf.glorot_normal_initializer(seed = 1))
        bias = tf.get_variable('bias_1',
                                shape = [16],
                                dtype = tf.float32,
                                initializer = tf.glorot_normal_initializer(seed = 1))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(image_batch, weights, strides = [1,1,1,1], padding = 'SAME'), bias)
        layer1 = tf.nn.relu(conv1, name = scope.name)

    with tf.variable_scope('maxpool_1') as scope:
        pool1 = tf.nn.max_pool(layer1, [1,2,2,1], strides = [1,1,1,1], padding='SAME')

    with tf.variable_scope('Conv_2') as scope:
        weights = tf.get_variable('weights_2',
                                    shape = [3,3,16,32],
                                    dtype = tf.float32,
                                    initializer = tf.glorot_normal_initializer(seed = 1))
        bias = tf.get_variable('bias_2',
                                shape = [32],
                                dtype = tf.float32,
                                initializer = tf.glorot_normal_initializer(seed = 1))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, weights, strides = [1,1,1,1], padding = 'SAME'), bias)
        layer2 = tf.nn.relu(conv2, name = scope.name)

    with tf.variable_scope('maxpool_2' ) as scope:
        pool2 = tf.nn.max_pool(layer2, [1,2,2,1], [1,1,1,1], padding='SAME',name='pool_2')

    with tf.variable_scope('Dense_1') as scope:
        flatten = tf.reshape(pool2,shape = [batch_size, -1], name='flatten')
        dim = flatten.get_shape()[1].value
        weights = tf.get_variable('dense_weights_1',
                                    shape = [dim, 128],
                                    dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer(seed = 1))
        bias = tf.get_variable('dense_bias', shape = [128], initializer=tf.glorot_normal_initializer(seed = 1))

        dense = tf.nn.relu(tf.add(tf.matmul(flatten, weights), bias), name = scope.name)

    with tf.variable_scope('softmax') as scope:
        weights  = tf.get_variable('dense_2',
                                    shape = [128, num_classes],
                                    dtype=tf.float32 ,
                                    initializer = tf.glorot_normal_initializer(seed = 1))

        bias = tf.get_variable('bias',
                                shape = [num_classes],
                                dtype=tf.float32,
                                initializer = tf.glorot_normal_initializer(seed = 1))
        dense_2 = tf.nn.softmax(tf.add(tf.matmul(dense, weights), bias), name = scope.name)

    return dense_2

def softmax_loss(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def optimizer(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_Step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op

def evaluate(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
