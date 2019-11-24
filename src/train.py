import tensorflow as tf
import numpy as np
from tqdm import tqdm
import dataset
import model


BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_CLASSES = 10
EPOCHS = 30

train_datset, test_dataset = dataset.create_dataset(BATCH_SIZE, EPOCHS)
