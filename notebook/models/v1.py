from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, DepthwiseConv2D, SeparableConv2D
from utils.utils import _fit_eval
import tensorflow as tf


#Tentativa inicial com 2 camadas convolucionais e 2 densas
def get_model(img_rows, img_cols,num_classes,x,y,x_valid=None,y_valid=None):
    
    model = Sequential()

    model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    tf.function(_fit_eval(model, x, y, x_valid, y_valid), jit_compile=True, experimental_follow_type_hints=True)
    
    return model