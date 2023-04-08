from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.io import imread
from skimage.util import img_as_float, apply_parallel
from skimage.util.shape import view_as_blocks
from skimage.transform import resize
from scipy.fft import dctn, idctn
import numpy as np
import matplotlib.pyplot as plt

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

def encode_channel(channel, chunk_size, snake_r, snake_c):
    channel_height, channel_width = channel.shape
    assert channel_height % chunk_size == 0
    assert channel_width % chunk_size == 0
#    channel = channel.reshape((-1, chunk_size, channel_height//chunk_size, chunk_size))
#    channel = np.moveaxis(channel, (0, 1, 2, 3), (0, 2, 1, 3))
    
#    channel = np.moveaxis(channel, (0, 1, 2, 3), (1, 2, 0, 3))
    channel = view_as_blocks(channel, (chunk_size, chunk_size))
    channel = dctn(channel, axes=(2,3), norm='ortho')
    channel = channel[:, :, snake_r, snake_c]
    return channel

def encode(image, chunk_size, snake_size):
    ycbcr = rgb2ycbcr(image)
    y, cb, cr = ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]

    snake_r, snake_c = snake(snake_size)
    cb_prime = encode_channel(cb, chunk_size, snake_r, snake_c)
    cr_prime = encode_channel(cr, chunk_size, snake_r, snake_c)

    return y, cb_prime, cr_prime

def decode_channel(encoded, chunk_size, snake_r, snake_c):
    channel_height_chunks, channel_width_chunks, _ = encoded.shape
    channel = np.zeros((channel_height_chunks, channel_width_chunks, chunk_size, chunk_size))
    channel[:, :, snake_r, snake_c] = encoded[:, :, :]
    channel = idctn(channel, axes=(2,3), norm='ortho')
    channel = np.moveaxis(channel, (0, 2, 1, 3), (0, 1, 2, 3))
    channel = np.reshape(channel, (channel_height_chunks*chunk_size, channel_width_chunks*chunk_size))
    return channel

def decode(y, cb_prime, cr_prime, chunk_size, snake_size):
    snake_r, snake_c = snake(snake_size)
    cb = decode_channel(cb_prime, chunk_size, snake_r, snake_c)
    cr = decode_channel(cr_prime, chunk_size, snake_r, snake_c)

    ycbcr = np.stack((y, cb, cr), axis=-1)
    return ycbcr2rgb(ycbcr)

image = imread('square.jpeg')
image = img_as_float(image)
image = resize(image, (400, 400, 3), anti_aliasing=True)

# parameters control color quality
chunk_size = 50
snake_size = 5

y, cb_prime, cr_prime = encode(image, chunk_size, snake_size)
image2 = decode(y, cb_prime, cr_prime, chunk_size, snake_size)

print('old color parameters', image[:,:,:2].size)
print('new color parameters', cb_prime.size+cr_prime.size)

f, (plot1, plot2) = plt.subplots(2,1) 
plot1.imshow(image)
plot2.imshow(image2)
plt.show()
