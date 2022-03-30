import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.python import keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from models.v4 import get_model

tf.config.optimizer.set_jit(True)

img_rows, img_cols = 28, 28
num_classes = 10


def data_prep(raw):
    out_y = keras.utils.np_utils.to_categorical(raw.label, num_classes)
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_file = "../data/train.csv"
test_file = "../data/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Split the data into training and validation sets
train_x, train_y = np.array(train_df.iloc[:, 1:]), np.array(train_df.iloc[:, 0])
x_all, y_all = data_prep(train_df)
x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.2, random_state=13)

get_model(img_rows, img_cols, num_classes, x_train, y_train, x_valid, y_valid) 