import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.compiler.xla import xla
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, DepthwiseConv2D, SeparableConv2D

from utils.utils import _fit_eval

def get_model(img_rows, img_cols,num_classes,x,y,x_valid=None,y_valid=None):
    model = Sequential()

    model.add(Conv2D(200, kernel_size=(2, 2),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))

    model.add(DepthwiseConv2D(kernel_size=(1, 5), strides=(1, 1), padding='same', depth_multiplier=1,
                              input_shape=(img_rows, img_cols, 1)))

    model.add(SeparableConv2D(filters=100, kernel_size=(4, 4),
                              activation='relu', padding='same'))

    model.add(Conv2D(50, kernel_size=(5, 1), activation='relu'))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    tf.function(_fit_eval(model, x, y, x_valid, y_valid), jit_compile=True, experimental_follow_type_hints=True)

    return model