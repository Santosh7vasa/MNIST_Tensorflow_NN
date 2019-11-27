import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

def create_dataset(batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.truediv(image, 255.0)
        return image, label
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(preprocess, num_parallel_calls = 10)
    train_dataset = train_dataset.repeat(epochs).shuffle(buffer_size = 50000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(preprocess, num_parallel_calls = 10)
    test_dataset = test_dataset.repeat(epochs).shuffle(buffer_size = 10000).batch(batch_size)
    return train_dataset, test_dataset

# with tf.Session() as sess:
#     data,_ = create_dataset(1,1)
#     iterator = data.make_one_shot_iterator()
#     image,label = iterator.get_next()
#     print(sess.run(image))
#     print(sess.run(label))
