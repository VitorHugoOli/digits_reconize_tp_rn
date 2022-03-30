
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.compiler.xla import xla
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, Masking

from utils.utils import _fit_eval

def get_model(img_rows, img_cols,num_classes,x,y,x_valid=None,y_valid=None):
    model = Sequential()


    model.add(Conv2D(200, kernel_size=(2, 2),
                    activation='relu',
                    input_shape=(img_rows, img_cols, 1)))
                    
    model.add(AveragePooling2D(pool_size=2))

    model.add(DepthwiseConv2D(kernel_size=(2, 2), strides=(1, 1), padding='same', depth_multiplier=1,
                            input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(100, kernel_size=(6, 6),
                    activation='relu',
                    padding='same'))

    model.add(Conv2D(num_classes, kernel_size=(8, 2),
                    activation='softmax',
                    padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Masking(mask_value=0))
    
    model.add(Dense(num_classes, activation='softmax'))
                    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tf.function(_fit_eval(model, x, y, x_valid, y_valid), jit_compile=True, experimental_follow_type_hints=True)




    return model