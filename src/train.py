import os
import sys
import argparse
import re
import os
import random
from datetime import datetime
import tensorflow as tf
from skimage.transform import resize
from skimage import img_as_float32
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from params import *
from util import *
    # TODO: rest of the U-net

    # TODO: final tanh? activation to limit output to (-0.5, 0.5) (actual tanh from -1 to 1)
    # TODO: 1% of the time during training, the full color is revealed to the network
    # TODO: patch size revealed to the network uniformly sampled from area 1-9, with average patch color 
    # TODO: number of points revealed geometric 1/8
    # TODO: patch location sampled from normal centered on image center
    # TODO: should make training data visualization
    # TODO: I improperly handled case where patches overlap

#     onv1-10. In conv1-4,
#    in every block, feature tensors are progressively halved spatially,
#    while doubling in the feature dimension. Each block contains 2-3
#    conv-relu pairs. In the second half, conv7-10, spatial resolution is
#    recovered, while feature dimensions are halved. In block conv5-6,
#    instead of halving the spatial resolution, dilated convolutions with
#    factor 2 is used.  Symmetric shortcut con-
#    nections are added to help the network recover spatial information
#     For example, the conv2 and conv3 blocks
#    are connected to the conv8 and conv9 blocks, respectively. Changes in spatial resolution are achieved using
#    subsampling or upsampling operations, and each convolution uses
#    a 3 × 3 kernel. BatchNorm layers are added after each convolutional
#    block, which has been shown to help training.
#    A subset of our network architecture, namely conv1-8 without
#    the shortcut connections, was used by Zhang et al. (2016). For these
#    layers, we fine-tune from these pre-trained weights. The added
#    conv9, conv10 layers and shortcut connections are trained from
#    scratch. A last conv layer, which is a 1 × 1 kernel, maps between
#    conv10 and the output color. Because the ab gamut is bounded, we
#    add a final tanh layer on the output, as is common practice when
#    generating images (Goodfellow et al. 2014; Zhu et al. 2016).
#

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '2', '3'],
        help='''Which task of the assignment to run -
        training a simple model (1), or training the entire model (2).''')

    return parser.parse_args()
ARGS = parse_args()

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

    x = x + x3

    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the 9th conv block
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding = "same", dilation_rate= 1, activation="relu")(x)
    x2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x2)

    x = x + x2

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #the last conv block
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding = "same", dilation_rate= 1, activation="relu")(x)
    x1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1, activation="relu")(x1)

    x = x + x1

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
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2,2), padding = "same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    #last block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding = "same", dilation_rate= 1)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(2, kernel_size=(1, 1), strides=(1,1), padding = "valid", dilation_rate= 1)(x)
    x = half_tanh_activation(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

def half_tanh_activation(x):
    '''
    returns - [-0.5, 0.5]
    '''
    return tf.tanh(x) / 2.0


model = None
if ARGS.task == "1":
    model = create_model()
elif ARGS.task == '2':
    model = create_model_simple()
elif ARGS.task == '3':
    model = create_model_simplest() 
else:
    raise ValueError('bad model number')

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    run_eagerly=True,
    loss=tf.keras.losses.Huber())

model.summary()

load_weights_from = None
if load_weights_from is not None:
    model.load_weights(load_weights_from)

training_data = tf.data.Dataset.load(training_path)
validation_data = tf.data.Dataset.load(validation_path)

model.fit(
    x=training_data,
    validation_data=validation_data,
    epochs=epochs_count,
    batch_size=None,           
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir='log/',
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='check/{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True),
    ],
    )
if ARGS.task == "3":
    model1 = create_model_simplest()

    model1.compile(
    optimizer=tf.keras.optimizers.Adam(),
    run_eagerly=True,
    loss=tf.keras.losses.Huber())

    model1.summary()

    model = model1
    model.fit(
    x=training_data,
    validation_data=validation_data,
    epochs=epochs_count,
    batch_size=None,           
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir='log/',
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='check/{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True),
    ],
    )
