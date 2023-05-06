from keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate, BatchNormalization
from keras.models import Model
from keras.layers.core import RepeatVector


def inception_model():
    embed_input = Input(shape=(1000,))

    #Encoder
    e_in = Input(shape=(256, 256, 1,))
    e = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(e_in)
    e = Conv2D(128, (3,3), activation='relu', padding='same')(e)
    e = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(e)
    e = Conv2D(256, (3,3), activation='relu', padding='same')(e)
    e = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(e)
    e = Conv2D(512, (3,3), activation='relu', padding='same')(e)
    e = Conv2D(512, (3,3), activation='relu', padding='same')(e)
    e = Conv2D(256, (3,3), activation='relu', padding='same')(e)

    #Fusion
    f = RepeatVector(32 * 32)(embed_input) 
    f = Reshape(([32, 32, 1000]))(f)
    f = concatenate([e, f], axis=3) 
    f = Conv2D(256, (1, 1), activation='relu', padding='same')(f) 

    #Decoder
    d = Conv2D(128, (3,3), activation='relu', padding='same')(f)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (3,3), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (3,3), activation='relu', padding='same')(d)
    d = Conv2D(16, (3,3), activation='relu', padding='same')(d)
    d = Conv2D(2, (3, 3), activation='tanh', padding='same')(d)
    d = UpSampling2D((2, 2))(d)

    model = Model(inputs=[e_in, embed_input], outputs=d)
    return model