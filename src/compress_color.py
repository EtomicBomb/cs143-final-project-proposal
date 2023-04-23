from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.io import imread
from skimage.util import img_as_float, apply_parallel
from skimage.util.shape import view_as_blocks
from skimage.transform import resize
from scipy.fft import dctn, idctn
import numpy as np
import matplotlib.pyplot as plt
import glob

def get_quantization(snake_len):
    quant_epsilon = 0.0001
    quantization = np.full((1, 2*snake_len), quant_epsilon)
    quantization[:, :4] += np.reshape(np.flip(np.square(np.arange(4))), (1, -1))
    print(quantization)
    return quantization

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

def lerp(x, in_min, in_max, out_min, out_max):
    ret = out_min + (x-in_min) * (out_max-out_min) / (in_max-in_min)
    return np.clip(ret, out_min, out_max)

def encode_whole(image, snake_len):
    ycbcr = rgb2ycbcr(image)
    y, color = ycbcr[:,:,0], ycbcr[:,:,1:]
    color = lerp(color, 16, 240, -1, 1)
    snake_r, snake_c = snake(snake_len)
    color_prime = dctn(color, axes=(0,1), norm='forward')[snake_r, snake_c, :]
#    color_prime = dctn(color, axes=(0,1), norm='ortho')[snake_r, snake_c, :]

    return y, color_prime

def decode_whole(y, color_prime, snake_len):
    height, width = y.shape
    snake_r, snake_c = snake(snake_len)

    color = np.zeros((height, width, 2))
    color[snake_r,snake_c,:] = color_prime[:,:]
    color = idctn(color, axes=(0,1), norm='forward')
#    color = idctn(color, axes=(0,1), norm='ortho')
    color = lerp(color, -1, 1, 16, 240)
    return ycbcr2rgb(np.dstack((y, color)))

def get_dataset(path, snake_len, mode, chunk_size):
    '''
    Produces inputs = (image_count, height, width) and outputs = (image_count, snake_len, 2)
    for mode == 'whole'
    or outputs = (image_count, height_chunks, width_chunks, 2, snake_len, 1) for mode == 'chunk'
    '''
    inputs = []
    outputs = []
    for (i, raw_image_path) in enumerate(glob.glob(path)):
        image = imread(raw_image_path)
        image = img_as_float(image)
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            # TODO: remove these from thumb
            continue

        # TODO: consider resize greyscale separately?
        # TODO: consider center crop to square before resize?
        
        y, color_prime = None, None
        if mode == 'whole':
            y, color_prime = encode_whole(image, snake_len)
        elif mode == 'chunk':
            y, color_prime = encode_chunk(image, chunk_size, snake_len)
        else:
            raise ValueError('bad mode ' + mode)

        inputs.append(y)
        outputs.append(color_prime)

    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    
    return inputs, outputs

# mogrify -path 'thumb' -resize '100x100^' -gravity center -extent 100x100 -strip 'raw/*'
if __name__ == '__main__':

    inputs, outputs = get_dataset('thumb/*', 500, 'whole', None) 
    #inputs, outputs = get_dataset('thumb/*', 10, 'chunk', 40) 
    np.save('inputs', inputs)
    np.save('outputs', outputs)

## parameters control color quality
#
#image = img_as_float(imread('raw/11746452_5bc1749a36.jpg'))
#h, w, _ = image.shape
#image = resize(image, (h-h%chunk_size, w-w%chunk_size, 3))
#
#y, color_prime = encode_whole(image, snake_len)
#image2 = decode_whole(y, color_prime, snake_len)
##y, color_prime = encode_chunk(image, chunk_size, snake_len)
##image2 = decode_chunk(y, color_prime, chunk_size, snake_len)
#
#print('old color parameters', image[:,:,:2].size)
#print('new color parameters', color_prime.size)
#
#f, (plot1, plot2) = plt.subplots(2,1) 
#plot1.imshow(image)
#plot2.imshow(image2)
#plt.show()
