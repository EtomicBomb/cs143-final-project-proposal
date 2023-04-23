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

def create_model():
    grey_in = tf.keras.Input(shape=(None, None, 1))
    hint_mask_in = tf.keras.Input(shape=(None, None, 1))
    hint_color_in = tf.keras.Input(shape=(None, None, 2))

    x = tf.concat((grey_in, hint_mask_in, hint_color_in), axis=-1)

    # TODO: rest of the U-net
    x = tf.keras.layers.Conv2DTranspose(2, kernel_size=(1, 1), strides=(1,1))(x)

    x = tf.keras.Model(inputs=(grey_in, hint_mask_in, hint_color_in), outputs=x)
    return x

model = create_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    run_eagerly=True,
    loss=tf.keras.losses.MeanSquaredError())

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
