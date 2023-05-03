
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
from inception_util import create_inception_embedding, preprocess_image, get_top_5_colors
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import os
import inception_params as hp

 # Test model
# test_data=tfds.load("imagenette", split="validation")
# test_data = test_data.map(preprocess_image)
# test_data = test_data.take(1)
# color_me_this = []
# for i in test_data:
#     color_me_this.append(img_to_array(i))




def get_suggested_colors(image):
    model = keras.models.load_model('inception_model.h5')
    target_shape = (hp.img_size, hp.img_size, 3)
    # image = img_to_array(image)
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


    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me_this[0][:,:,0]
    cur[:,:,1:] = output[0]
    colorized = lab2rgb(cur)
    imsave("result/img_"+str(0)+".png", colorized)
    print(get_top_5_colors(cur))

    return get_top_5_colors(cur)

# for filename in os.listdir('Test/'):
#     img = load_img('Test/' + filename)
#     target_shape = (hp.img_size, hp.img_size, 3)
#     img = tf.image.resize(img, target_shape[:2])    
#     get_suggested_colors(img)

# img = colorized







