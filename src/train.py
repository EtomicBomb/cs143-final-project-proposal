import os
import sys
import argparse
import re
import os
import random
from datetime import datetime
import tensorflow as tf
import numpy as np
import tensorflow as tf

from params import *
from util import *

'''
'''

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        help='''Which task of the assignment to run -
        training a simple model (1), or training the entire model (2).''')

    return parser.parse_args()

def half_tanh_activation(x):
    '''
    returns - [-0.5, 0.5]
    '''
    return tf.tanh(x) / 2.0

def create_model():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    #1st conv block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x1 = x

    #second conv block
    x = x[:,:,::2,::2]
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x2 = x

    #3rd conv block
    x = x[:,:,::2,::2]
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x3 = x

    #the 4th conv block
    x = x[:,:,::2,::2]
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 5th conv block
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 6th conv block
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 7th conv block
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 8th conv block
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2,2), padding = "same", dilation_rate= 1, activation="relu")(x)
    x3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x3)

#    x = x + x3
    x = tf.concat((x, x3), axis=-1)

    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 9th conv block
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding = "same", dilation_rate= 1, activation="relu")(x)
    x2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x2)

    x = tf.concat((x, x2), axis=-1)
#    x = x + x2

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the last conv block
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding = "same", dilation_rate= 1, activation="relu")(x)
    x1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x1)

#    x = x + x1
    x = tf.concat((x, x1), axis=-1)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(2, kernel_size=(1, 1), strides=(1,1), padding = "valid", dilation_rate= 1)(x)
    x = half_tanh_activation(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x


def create_model_simple():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    #first block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #second block
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #third block
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #fourth block
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2,2), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #fifth block
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #sixth block
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2,2), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #last block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(2, kernel_size=(1, 1), strides=(1,1), padding = "valid", dilation_rate= 1)(x)
    x = half_tanh_activation(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def create_model_simplest():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    #first block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #sixth block
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1,1), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #last block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(2, kernel_size=(1, 1), strides=(1,1), padding = "valid", dilation_rate= 1)(x)
    x = half_tanh_activation(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def create_model_simplest2():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    x = tf.keras.layers.Conv2D(2, kernel_size=1, strides=1, padding = "valid")(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def create_model_cool():
    # https://www.mdpi.com/2673-2688/1/4/29/htm
    # https://www.mdpi.com/ai/ai-01-00029/article_deploy/html/images/ai-01-00029-g002.png

    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))
    valid = tf.ones_like(grey_in)
    x = tf.concat((grey_in, hint_mask_in, hint_color_in, valid), axis=-1)

    # blue
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
#    x1 = x
    
    # orange
    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)

    # blue
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x2 = x

    # orange
    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)

    # blue
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x3 = x

    # orange
    x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)

    # blue
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x4 = x

    # orange
    x = tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)

    # blue
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)


    # green
    x = tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

#    x = tf.concat((x, x4), axis=-1)
    # blue
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

    # green
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.concat((x, x3), axis=-1)
    # blue
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

    # green
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

#    x = tf.concat((x, x2), axis=-1)
    # blue
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

    # green
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

#    x = tf.concat((x, x1), axis=-1)
    # blue
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=32)(x)
    x = tf.keras.layers.ReLU()(x)

    # navy
    x = tf.keras.layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(x)
    x = tf.math.sigmoid(x) - 0.5 # output in [-0.5, 0.5]

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def create_model_encoder():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))
    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    slope = 0.1

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)

    # down
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # continue
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)

    # continue
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)

    # continue
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)

    # up
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)
    
    # continue
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.ReLU(negative_slope=slope)(x)

    x = tf.keras.layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(x)
    x = tf.math.sigmoid(x) - 0.5 # output in [-0.5, 0.5]

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

if model_number == 0:
    model = tf.keras.models.load_model(load_weights_from)
elif model_number == 1:
    model = create_model()
elif model_number == 2:
    model = create_model_simple()
elif model_number == 3:
    model = create_model_simplest() 
elif model_number == 4:
    model = create_model_cool() 
elif model_number == 5:
    model = create_model_simplest2() 
elif model_number == 6:
    model = create_model_encoder() 
else:
    raise ValueError('bad model number ' + str(model_number))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    run_eagerly=True,
    loss=tf.keras.losses.Huber())

model.summary()

training_data = tf.data.Dataset.load(training_path)
validation_data = tf.data.Dataset.load(validation_path)

model.fit(
    x=training_data.batch(batch_size),
    validation_data=validation_data.batch(batch_size),
    epochs=epochs_count,
    batch_size=None,           
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir='log/',
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='check/{epoch:02d}-{val_loss:.5f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True),
    ],
)
