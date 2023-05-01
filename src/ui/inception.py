
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from inception_model import inception_model
from inception_util import create_inception_embedding, image_a_b_gen


# Get imagesx
target_shape = (256, 256, 3)


model=inception_model()


#Generate training data
batch_size = 10


#Train model      
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
model.fit(image_a_b_gen(batch_size), epochs=5, steps_per_epoch=1)
model.summary()


color_me = []
for filename in os.listdir('Test/'):
    img = load_img('Test/' + filename)
    resized_img = img.resize(target_shape[:2])
    array_img = img_to_array(resized_img)
    color_me.append(array_img)
color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))