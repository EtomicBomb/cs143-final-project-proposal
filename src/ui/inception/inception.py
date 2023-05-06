
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
from inception_util import create_inception_embedding, preprocess_image, get_top_5_colors
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import inception_params as hp
import cv2




def get_suggested_colors(image):
    model = keras.models.load_model('last_inception_model.h5')
    target_shape = (hp.img_size, hp.img_size, 3)

    # test_data=tfds.load("tf_flowers", split="train")
    # test_data = test_data.map(preprocess_image)
    # test_data = test_data.shuffle(buffer_size=1024)

    # test_data = test_data.take(10)
    # color_me_this=[]
    # for i in test_data:
    #     img = img_to_array(i)
    #     img = img.astype(np.uint8)
    #     gamma = 1.5
    #     inv_gamma = 1.0 / gamma
    #     lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    #     img = cv2.LUT(img, lut)
        # color_me_this.append(img)

    image = img_to_array(image)
    image = image.astype(np.uint8)
    # increase image brightness
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(image, lut)
    image = tf.image.resize(image, target_shape[:2])    
    color_me_this = []
    color_me_this.append(img_to_array(image))
    color_me_this = np.array(color_me_this, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me_this))
    color_me_embed = create_inception_embedding(gray_me)
    color_me_this = rgb2lab(1.0/255*color_me_this)[:,:,:,0]
    color_me_this = color_me_this.reshape(color_me_this.shape+(1,))


    # Test model
    output = model.predict([color_me_this, color_me_embed])
    output = output * 128


    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me_this[i][:,:,0]
        cur[:,:,1:] = output[i]
        colorized = lab2rgb(cur)
        colorized_uint8 = (255 * colorized).astype(np.uint8)
        imsave("result/img_"+str(i)+".png", colorized_uint8)

        return get_top_5_colors(cur)





def get_colorized_inception(image):
    model = keras.models.load_model('last_inception_model.h5')
    target_shape = (hp.img_size, hp.img_size, 3)

    image = img_to_array(image)
    ori_shape=image.shape[:2]
    image = image.astype(np.uint8)
    # increase image brightness
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(image, lut)
    image = tf.image.resize(image, target_shape[:2])    
    color_me_this = []
    color_me_this.append(img_to_array(image))
    color_me_this = np.array(color_me_this, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me_this))
    color_me_embed = create_inception_embedding(gray_me)
    color_me_this = rgb2lab(1.0/255*color_me_this)[:,:,:,0]
    color_me_this = color_me_this.reshape(color_me_this.shape+(1,))


    # Test model
    output = model.predict([color_me_this, color_me_embed])
    output = output * 128


    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me_this[i][:,:,0]
        cur[:,:,1:] = output[i]
        colorized = lab2rgb(cur)
        colorized_uint8 = (255 * colorized).astype(np.uint8)
        colorized_uint8=tf.image.resize(colorized_uint8, ori_shape) 
        # imsave("result/img_"+str(i)+".png", colorized_uint8)

        return colorized_uint8


