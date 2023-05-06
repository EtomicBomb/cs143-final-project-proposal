import tensorflow as tf
from inception_model import inception_model
from inception_util import image_a_b_gen
import inception_params as hp
import sys
from tensorflow import keras


checkpoint_filepath = './checkpoint/weights.{epoch:02d}-{accuracy:.2f}.h5'
if len(sys.argv) > 1:
    check_file= './checkpoint/' + sys.argv[1];
    model = keras.models.load_model(check_file)
    initial_epoch = int(sys.argv[2])
else:
    model = inception_model()
    initial_epoch = 0


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='accuracy',
    verbose=1,
    mode='max',
    save_best_only=False,
    period=50)


model=inception_model()

#Train model      
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate), loss='mse', metrics=['accuracy'])
model.fit(image_a_b_gen(hp.batch_size), epochs=hp.num_epochs, steps_per_epoch=hp.steps_per_epoch,callbacks=[model_checkpoint_callback],initial_epoch=initial_epoch)
model.summary()
model.save("inception_model.h5")







