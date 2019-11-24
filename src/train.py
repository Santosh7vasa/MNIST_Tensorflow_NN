import tensorflow as tf
import numpy as np
from tqdm import tqdm
import dataset
import model


BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_CLASSES = 10
EPOCHS = 30
sess = tf.Session()
train_datset, test_dataset = dataset.create_dataset(BATCH_SIZE, EPOCHS)
iterator = train_datset.make_initializable_iterator()
#sess.run(iterator.initializer)
X,Y = iterator.get_next()

logits = model.inference(X, BATCH_SIZE, NUM_CLASSES)
loss = model.softmax_loss(logits, Y)
train_op = model.optimizer(loss, LEARNING_RATE)
accuracy = model.evaluate(logits,Y)

init = tf.global_variables_initializer()
sess.run(init)

for step in tqdm(range(0, (60000*30)/32)):
    sess.run(train_op)

    if step % 100 == 0 or step == 1:
        # Calculate batch loss and accuracy
        # (note that this consume a new batch of data)
        loss, acc = sess.run([loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))
