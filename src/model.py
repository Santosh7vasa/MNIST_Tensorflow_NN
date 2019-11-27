import tensorflow as tf
import numpy as np
from tqdm import tqdm
from dataset import create_dataset


def inference(image_batch, batch_size, num_classes):
    with tf.variable_scope('Conv_1') as scope:
        weights_1 = tf.get_variable('weights_1',
                                    shape = [3,3,1,16],
                                    dtype = tf.float32,
                                    initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias_1 = tf.get_variable('bias_1',
                                shape = [16],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        image_batch = tf.reshape(image_batch, shape = [-1, 28, 28, 1])
        conv1 = tf.nn.bias_add(tf.nn.conv2d(image_batch, weights_1, strides = [1,1,1,1], padding = 'SAME'), bias_1)
        layer1 = tf.nn.relu(conv1, name = scope.name)

    with tf.variable_scope('maxpool_1') as scope:
        pool1 = tf.nn.max_pool(layer1, [1,2,2,1], strides = [1,2,2,1], padding='VALID')

    with tf.variable_scope('Conv_2') as scope:
        weights_2 = tf.get_variable('weights_2',
                                    shape = [3,3,16,32],
                                    dtype = tf.float32,
                                    initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias_2 = tf.get_variable('bias_2',
                                shape = [32],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, weights_2, strides = [1,1,1,1], padding = 'SAME'), bias_2)
        layer2 = tf.nn.relu(conv2, name = scope.name)

    with tf.variable_scope('maxpool_2' ) as scope:
        pool2 = tf.nn.max_pool(layer2, [1,2,2,1], [1,2,2,1], padding='VALID',name='pool_2')

    with tf.variable_scope('Dense_1') as scope:
        flatten = tf.reshape(pool2,shape = [batch_size, -1], name='flatten')
        dim = pool2.get_shape()[1].value * pool2.get_shape()[3].value * pool2.get_shape()[2].value
        weights_3 = tf.get_variable('dense_weights_1',
                                    shape = [dim, 128],
                                    dtype=tf.float32,
                                    initializer= tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        bias_3 = tf.get_variable('dense_bias', shape = [128], initializer= tf.constant_initializer(0.1))

        dense = tf.nn.relu(tf.add(tf.matmul(flatten, weights_3), bias_3), name = scope.name)

    with tf.variable_scope('softmax') as scope:
        weights_4  = tf.get_variable('dense_2',
                                    shape = [128, num_classes],
                                    dtype=tf.float32 ,
                                    initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

        bias_4 = tf.get_variable('bias',
                                shape = [num_classes],
                                dtype=tf.float32,
                                initializer = tf.constant_initializer(0.1))
        dense_2 = tf.nn.softmax(tf.add(tf.matmul(dense, weights_4), bias_4), name = scope.name)

    return dense_2

def softmax_loss(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(labels, depth = 10))
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def optimizer(loss, learning_rate):
    with tf.name_scope('optimizer'):
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
        # global_Step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = opt.minimize(loss)
    return train_op

def evaluate(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.cast(labels, tf.int32)
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy

train, test = create_dataset(32,30)
it_train = train.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(it_train.initializer)
    X, Y = it_train.get_next()
    print(X.shape, Y.shape)
    logits = inference(X,32,10)
    loss = softmax_loss(logits,Y)
    train_op = optimizer(loss, 0.001)
    accuracy = evaluate(logits, Y)
    sess.run(tf.global_variables_initializer())
    for step in range(0,600000):
        sess.run(train_op)
        if step%100 == 0 or step == 1:

            loss_, acc = sess.run([loss, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
      "{:.4f}".format(loss_) + ", Training Accuracy= " + \
      "{:.3f}".format(acc))
