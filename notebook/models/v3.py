import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
import tensorflow.python.keras.models as models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
import tensorflow as tf


from utils.utils import _fit_eval

def get_model(img_rows, img_cols,num_classes,x,y,x_valid=None,y_valid=None):

    img_rows, img_cols = 28, 28
    num_classes = 10

    model = models.Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adadelta', metrics=['accuracy'])

    tf.function(_fit_eval(model, x, y, x_valid, y_valid), jit_compile=True, experimental_follow_type_hints=True)


    return model


