from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, DepthwiseConv2D, SeparableConv2D, MaxPooling2D
import tensorflow as tf

from utils.utils import _fit_eval

def get_model(img_rows, img_cols,num_classes,x,y,x_valid=None,y_valid=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                     input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    tf.function(_fit_eval(model, x, y, x_valid, y_valid), jit_compile=True, experimental_follow_type_hints=True)


    return model