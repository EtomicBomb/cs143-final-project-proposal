#!/usr/bin/env python
#coding: utf8

import cv2 
import time
import math
import pickle
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float32, img_as_ubyte
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale

from compress_color import *


value = {'a': 0}
cluster_centers = None

snake_len = 30
quantization = get_quantization(snake_len)

def update_value(v):
    value['a'] = v

def bin_snake(snake):
    snake = np.ravel(snake) * quantization
    difference = np.sum(np.square(cluster_centers - snake), axis=1)
    best_bin = np.argmin(difference)
    
    return best_bin

def unbin_snake(cluster_index):
    cluster_center = cluster_centers[cluster_index,:] / quantization
    return np.reshape(cluster_center, (snake_len, 2))

def my_get_images():
    vc = cv2.VideoCapture(0)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def read_webcam():
        image = None
        while image is None:
            _, image = vc.read()

        image = cv2.imread('/home/ethan/Pictures/flower2.jpeg')
        image = img_as_float32(image)

#        if image.shape[1] > image.shape[0]:
#            cropx = int((image.shape[1]-image.shape[0])/2)
#            cropy = 0
#        elif image.shape[0] > image.shape[1]:
#            cropx = 0
#            cropy = int((image.shape[0]-image.shape[1])/2)

#        image = image[cropy:image.shape[0]-cropy, cropx:image.shape[1]-cropx,:]

        return image

    def get_images_ret():
        image = read_webcam()
        snake_len = 30
        y, color_prime = encode_whole(image, snake_len)
        image2 = decode_whole(y, color_prime, snake_len) 

        color_prime_unbinned = unbin_snake(bin_snake(color_prime))
        image3 = decode_whole(y, color_prime_unbinned, snake_len)

#        snake_len = 4
#        chunk_size = 20
#        y, color_prime = encode_chunk(image, chunk_size, snake_len)
#        image2 = decode_chunk(y, color_prime, chunk_size, snake_len)
        return np.hstack((image, image2, image3))
        
    return get_images_ret


def feed(get_images):

    cv2.namedWindow("Live Demo", 0)
    cv2.createTrackbar('slider', "Live Demo", 0, 800, update_value)

    # Main loop
    while True:
        start = time.perf_counter()

        output = get_images()
        cv2.resizeWindow("Live Demo", output.shape[1]*2, output.shape[0]*2)
        output = np.clip(output*255, 0, 255)
        cv2.imshow("Live Demo", output.astype(np.uint8))
        cv2.waitKey(1)

#        print('framerate = {} fps \r'.format(1. / (time.perf_counter() - start)))
    


if __name__ == '__main__':
#    cluster_centers = pickle.load(open("cluster-centers.pkl", "rb"))
    cluster_centers = np.load('cluster-centers.npy')
    feed(my_get_images())
