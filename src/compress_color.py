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

def encode_chunk(image, chunk_size, snake_len):
    ycbcr = rgb2ycbcr(image)
    y, color_prime = ycbcr[:,:,0], ycbcr[:,:,1:]

    snake_r, snake_c = snake(snake_len)

    color_prime = view_as_blocks(color_prime, (chunk_size, chunk_size, 1))
    color_prime = dctn(color_prime, axes=(3,4), norm='ortho')
    color_prime = color_prime[:, :, :, snake_r, snake_c, :]

    return y, color_prime

def decode_chunk(y, color_prime, chunk_size, snake_len):
    snake_r, snake_c = snake(snake_len)
    channel_height_chunks, channel_width_chunks, _, _, _ = color_prime.shape
    color = np.zeros((channel_height_chunks, channel_width_chunks, 2, chunk_size, chunk_size, 1))
    color[:, :, :, snake_r, snake_c, :] = color_prime
    color = idctn(color, axes=(3,4), norm='ortho')
    color = np.moveaxis(color, (0, 3, 5, 1, 4, 2), (0, 1, 2, 3, 4, 5))
    color = np.reshape(color, (channel_height_chunks*chunk_size, channel_width_chunks*chunk_size, 2))

    return ycbcr2rgb(np.dstack((y, color)))

def encode_whole(image, snake_len):
    ycbcr = rgb2ycbcr(image)
    y, color = ycbcr[:,:,0], ycbcr[:,:,1:]

    snake_r, snake_c = snake(snake_len)
    color_prime = dctn(color, axes=(0,1), norm='ortho')[snake_r, snake_c, :]

    return y, color_prime

def decode_whole(y, color_prime, snake_len):
    height, width = y.shape
    snake_r, snake_c = snake(snake_len)

    color = np.zeros((height, width, 2))
    color[snake_r,snake_c,:] = color_prime[:,:]
    color = idctn(color, axes=(0,1), norm='ortho')

    return ycbcr2rgb(np.dstack((y, color)))

# parameters control color quality
chunk_size = 500
snake_len = 20

image = img_as_float(imread('square3.jpeg'))
h, w, _ = image.shape
image = resize(image, (h-h%chunk_size, w-w%chunk_size, 3))

#y, color_prime = encode_whole(image, snake_len)
#image2 = decode_whole(y, color_prime, snake_len)
y, color_prime = encode_chunk(image, chunk_size, snake_len)
image2 = decode_chunk(y, color_prime, chunk_size, snake_len)

print('old color parameters', image[:,:,:2].size)
print('new color parameters', color_prime.size)

f, (plot1, plot2) = plt.subplots(2,1) 
plot1.imshow(image)
plot2.imshow(image2)
plt.show()
