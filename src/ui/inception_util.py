
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import img_to_array, load_img


inception = InceptionV3(weights='imagenet', include_top=True)
datagen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,rotation_range=20,horizontal_flip=True)



target_shape = (256, 256, 3)
X = []
for filename in os.listdir('Train/'):
    img = load_img('Train/' + filename)
    resized_img = img.resize(target_shape[:2])
    array_img = img_to_array(resized_img)
    X.append(array_img)
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X

def create_inception_embedding(grayscaled_rgb):

    #Load weights
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception(grayscaled_rgb_resized)
    return embed


def image_a_b_gen(batch_size):
    # Image transformer
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch,embed], Y_batch)