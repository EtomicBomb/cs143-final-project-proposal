
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
import tensorflow as tf
from inception_model import inception_model
from inception_util import create_inception_embedding, image_a_b_gen, preprocess_image, get_top_5_colors
import tensorflow_datasets as tfds



batch_size = 16
num_epochs = 30

model=inception_model()


#Train model      
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse', metrics=['accuracy'])
model.fit(image_a_b_gen(batch_size), epochs=num_epochs, steps_per_epoch=1)
model.summary()
# model.save("my_model")

# Test model
test_data=tfds.load("imagenette", split="validation")
test_data = test_data.map(preprocess_image)
test_data = test_data.take(1)
color_me_this = []
for i in test_data:
    color_me_this.append(img_to_array(i))


color_me_this = np.array(color_me_this, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me_this))
color_me_embed = create_inception_embedding(gray_me)
color_me_this = rgb2lab(1.0/255*color_me_this)[:,:,:,0]
color_me_this = color_me_this.reshape(color_me_this.shape+(1,))


# Test model
output = model.predict([color_me_this, color_me_embed])
output = output * 128

# Output colorizations
# for i in range(len(output)):
#     cur = np.zeros((256, 256, 3))
#     cur[:,:,0] = color_me_this[i][:,:,0]
#     cur[:,:,1:] = output[i]
#     colorized = lab2rgb(cur)
#     imsave("result/img_"+str(i)+".png", colorized)


cur = np.zeros((256, 256, 3))
cur[:,:,0] = color_me_this[0][:,:,0]
cur[:,:,1:] = output[0]
colorized = lab2rgb(cur)
imsave("result/img_"+str(i)+".png", colorized)
get_top_5_colors(cur)

# img = colorized







