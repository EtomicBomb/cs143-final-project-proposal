from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.io import imread
from skimage.util import img_as_float, apply_parallel
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

def snake(size):
    indices = np.zeros((2, size), dtype=int)
    xy = np.array([0, 0])
    dxdy = np.array([-1, 1])
    reset = np.array([0, 1])
    diag_index = 1
    i = 0
    while True:
        for _ in range(diag_index):
            if i >= size:
                return indices
            indices[:, i] = xy
            xy += dxdy
            i += 1
        xy = diag_index * reset
        diag_index += 1
        # zigzag
        reset = np.flip(reset)
        dxdy = np.flip(dxdy)

def encode_channel(channel, chunk_height, chunk_width, snake_r, snake_c):
    channel_height, channel_width = channel.shape
    assert channel_height % chunk_height == 0
    assert channel_width % chunk_width == 0
    channel = channel.reshape((-1, chunk_height, channel_height//chunk_height, chunk_width))
    channel = np.moveaxis(channel, (0, 1, 2, 3), (0, 2, 1, 3))
    channel = dctn(channel, axes=(2,3), norm='ortho')
    channel = channel[:, :, snake_r, snake_c]
    return channel

def encode(image, chunk_height, chunk_width, snake_size):
    y, cb, cr = np.split(rgb2ycbcr(image), 3, axis=2)
    y = np.squeeze(y)
    cb = np.squeeze(cb)
    cr = np.squeeze(cr)

    snake_r, snake_c = snake(snake_size)
    cb_prime = encode_channel(cb, chunk_height, chunk_width, snake_r, snake_c)
    cr_prime = encode_channel(cr, chunk_height, chunk_width, snake_r, snake_c)

    return y, cb_prime, cr_prime

def decode_channel(encoded, chunk_height, chunk_width, snake_r, snake_c):
    channel_height_chunks, channel_width_chunks, _ = encoded.shape
    channel = np.zeros((channel_height_chunks, channel_width_chunks, chunk_height, chunk_width))
    channel[:, :, snake_r, snake_c] = encoded[:, :, :]
    channel = idctn(channel, axes=(2,3), norm='ortho')
    channel = np.moveaxis(channel, (0, 2, 1, 3), (0, 1, 2, 3))
    channel = np.reshape(channel, (channel_height_chunks*chunk_height, channel_width_chunks*chunk_width))
    return channel

def decode(y, cb_prime, cr_prime, chunk_height, chunk_width, snake_size):
    snake_r, snake_c = snake(snake_size)
    cb = decode_channel(cb_prime, chunk_height, chunk_width, snake_r, snake_c)
    cr = decode_channel(cr_prime, chunk_height, chunk_width, snake_r, snake_c)

    print(y.shape, cb.shape, cr.shape)
    ycbcr = np.stack((y, cb, cr), axis=-1)
    return ycbcr2rgb(ycbcr)

image = imread('input/square.jpeg')
image = img_as_float(image)
image = resize(image, (400, 400, 3), anti_aliasing=True)

chunk_height = 50
chunk_width = 50
snake_size = 5
y, cb_prime, cr_prime = encode(image, chunk_height, chunk_width, snake_size)
image2 = decode(y, cb_prime, cr_prime, chunk_height, chunk_width, snake_size)

print('old color parameters', image[:,:,:2].size)
print('new color parameters', cb_prime.size+cr_prime.size)

f, (plot1, plot2) = plt.subplots(2,1) 
plot1.imshow(image)
plot2.imshow(image2)
plt.show()
