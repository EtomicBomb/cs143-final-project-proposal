
from skimage.io import imread
import glob
import numpy as np
from skimage.util import img_as_float
from compress_color import *
from sklearn.cluster import KMeans
import pickle

def patterns(path, snake_len):
    '''
    Produces inputs = (image_count, height, width) and outputs = (image_count, snake_len, 2)
    for mode == 'whole'
    or outputs = (image_count, height_chunks, width_chunks, 2, snake_len, 1) for mode == 'chunk'
    '''
    inputs = []
    for (i, raw_image_path) in enumerate(glob.glob(path)):
        image = imread(raw_image_path)
        image = img_as_float(image)
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            continue

        # TODO: randm crops

        y, color_prime = encode_whole(image, snake_len)
        inputs.append(color_prime)
        
    inputs = np.stack(inputs)

    inputs = np.reshape(inputs, (-1, 2*snake_len))
    quantization = get_quantization(snake_len)
    inputs = inputs * quantization

    print('running kmeans')
    kmeans = KMeans(n_clusters=3000, max_iter=10000, n_init=10, random_state=0).fit(inputs)
    return kmeans

if __name__ == '__main__':
    snake_len = 30
    kmeans = patterns('thumb/*', snake_len)
#    pickle.dump(clt, open("cluster-centers.pkl", "wb"))
    np.save('cluster-centers', kmeans.cluster_centers_)
