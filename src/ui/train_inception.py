import tensorflow as tf
from inception_model import inception_model
from inception_util import image_a_b_gen

import inception_params as hp


model=inception_model()

#Train model      
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate), loss='mse', metrics=['accuracy'])
model.fit(image_a_b_gen(hp.batch_size), epochs=hp.num_epochs, steps_per_epoch=hp.steps_per_epoch)
model.summary()
model.save("inception_model.h5")







