
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow_datasets as tfds
from sklearn.cluster import KMeans
from collections import Counter
import inception_params as hp



# X = []
# for filename in os.listdir('Train/'):
#     img = load_img('Train/' + filename)
#     resized_img = img.resize(target_shape[:2])
#     array_img = img_to_array(resized_img)
#     X.append(array_img)
# X = np.array(X, dtype=float)


inception = InceptionResNetV2(weights='imagenet', include_top=True)
datagen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,rotation_range=20,horizontal_flip=True, width_shift_range=0.2,
    height_shift_range=0.2,fill_mode='nearest')



def preprocess_image(img):
    target_shape = (hp.img_size, hp.img_size, 3)
    image = img['image']
    image = tf.image.resize(image, target_shape[:2])
    return image

def get_train_data():
    ds = tfds.load('imagenette', split='train')
    ds = ds.map(preprocess_image)

    # Create an array from the dataset
    X = []
    for i in ds:
        X.append(img_to_array(i))
    X = np.array(X, dtype=float)
    Xtrain = 1.0/255*X
    return Xtrain


def create_inception_embedding(grayscaled_rgb):
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
    for batch in datagen.flow(get_train_data(), batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch,embed], Y_batch)


def get_top_5_colors(lab_img):
    lab_img = lab_img.reshape((lab_img.shape[0] * lab_img.shape[1], 3))
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(lab_img)
    label_counts = Counter(labels)
    top_five_labels = label_counts.most_common(5)
    top_colors = [kmeans.cluster_centers_[l] for l, _ in top_five_labels]

    top_rgb_colors = []
    for color in top_colors:
        rgb = lab2rgb(np.reshape(color, (1, 1, 3)))
        rgb_color_int = np.round(rgb * 255).astype(int)
        top_rgb_colors.append(rgb_color_int)
    print(top_rgb_colors)
    return top_rgb_colors