
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
from inception_util import create_inception_embedding, preprocess_image, get_top_5_colors, adjust_gamma, lighten_rgb
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import inception_params as hp




def get_suggested_colors(image):
    model = keras.models.load_model('newest_inception_model.h5')
    target_shape = (hp.img_size, hp.img_size, 3)

    # test_data=tfds.load("tf_flowers", split="train")
    # test_data = test_data.map(preprocess_image)
    # test_data = test_data.shuffle(buffer_size=1024)

    # test_data = test_data.take(10)
    # color_me_this=[]
    # for i in test_data:
    #     img = img_to_array(i)
    #     img = img.astype(np.uint8)
        # color_me_this.append(img)

    image = img_to_array(image)
    image = image.astype(np.uint8)
    # increase image brightness
    image=adjust_gamma(image)
    image = tf.image.resize(image, target_shape[:2])    

    color_me_this = []
    color_me_this.append(img_to_array(image))
    color_me_this = np.array(color_me_this, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me_this))
    color_me_embed = create_inception_embedding(gray_me)
    color_me_this = rgb2lab(1.0/255*color_me_this)[:,:,:,0]
    color_me_this = color_me_this.reshape(color_me_this.shape+(1,))


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
    model = keras.models.load_model('newest_inception_model.h5')
    target_shape = (hp.img_size, hp.img_size, 3)

    image = img_to_array(image)
    ori_shape=image.shape[:2]
    image = image.astype(np.uint8)
    # increase image brightness
    image=adjust_gamma(image)

    image = tf.image.resize(image, target_shape[:2])    
    color_me_this = []
    color_me_this.append(img_to_array(image))
    color_me_this = np.array(color_me_this, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me_this))
    color_me_embed = create_inception_embedding(gray_me)
    color_me_this = rgb2lab(1.0/255*color_me_this)[:,:,:,0]
    color_me_this = color_me_this.reshape(color_me_this.shape+(1,))

    output = model.predict([color_me_this, color_me_embed])
    output = output * 128


    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me_this[i][:,:,0]
        cur[:,:,1:] = output[i]
        colorized = lab2rgb(cur)
        colorized_uint8 = (255 * colorized).astype(np.uint8)
        colorized_uint8=tf.image.resize(colorized_uint8, ori_shape) 

        colorized_uint8 = tf.cast(colorized_uint8, tf.float32)
        colorized_uint8 = tf.reshape(colorized_uint8, [colorized_uint8.shape[0], colorized_uint8.shape[1], 3])
        colorized_uint8 = lighten_rgb(colorized_uint8, 1.0)
        colorized_uint8 = tf.cast(colorized_uint8, tf.uint8)
        imsave("result/img_"+str(i)+".png", colorized_uint8)

        return colorized_uint8


