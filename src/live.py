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

from diffusion import *
from compress_color import *

snake_len = 1000
t_max = 2*snake_len + snake_len
diffusion = Diffusion(0.95, 2*snake_len, t_max)

value = {'a': 0}

def update_value(v):
    value['a'] = v

def my_get_images():
    vc = cv2.VideoCapture(0)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def read_webcam():
        image = None
        while image is None:
            _, image = vc.read()

        image = cv2.imread('/home/ethan/Pictures/flower.jpeg')
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

        y, color_prime = encode_whole(image, snake_len)


        color_prime_prime = np.reshape(diffusion.sample(np.ravel(color_prime), value['a']), (-1, 2))
        print(np.min(color_prime), np.max(color_prime), np.min(color_prime_prime), np.max(color_prime_prime))

        image2 = decode_whole(y, color_prime_prime, snake_len) 

#        snake_len = 4
#        chunk_size = 20
#        y, color_prime = encode_chunk(image, chunk_size, snake_len)
#        image2 = decode_chunk(y, color_prime, chunk_size, snake_len)
        return np.hstack((image, image2))
        
    return get_images_ret


def feed(get_images):
    cv2.namedWindow("Live Demo", 0)
    cv2.createTrackbar('slider', "Live Demo", 0, t_max-1, update_value)

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
    feed(my_get_images())
