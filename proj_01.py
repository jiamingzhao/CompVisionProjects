# -*- coding: utf-8 -*-
import skimage, skimage.io, pylab, scipy.ndimage.filters
import numpy as np


def main():
    canny_edge()

    
def canny_edge():
    orig_img = skimage.io.imread('./flower.jpg')  # read image
    orig_img=skimage.img_as_float(orig_img)       # convert image to float
    gr_img = luminance(orig_img)                  # call luminance subroutine for 2d array
    
    conv_g_img = scipy.ndimage.filters.convolve(gr_img, get_gaussian_filter())
    conv_sx_img = scipy.ndimage.filters.convolve(conv_g_img, get_sobel(1))
    conv_sy_img = scipy.ndimage.filters.convolve(conv_g_img, get_sobel(0))
    
    # pylab.imshow(conv_sy_img, cmap="gray")
    
    magnitude = np.sqrt( np.add( np.square(conv_sx_img), np.square(conv_sy_img) ) )
    pylab.imshow(magnitude, cmap="gray")

def get_sobel(x):
    if x == 1:
        filter = [ [1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1] ]
    else:
        filter = [ [1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1] ]
    return filter

def get_gaussian_filter():
    filter = [ [1, 4, 7, 4, 1],
               [4, 16, 26, 16, 4],
               [7, 26, 41, 26, 7],
               [4, 16, 26, 16, 4],
               [1, 4, 7, 4, 1] ]
            
    return filter


def luminance(img):
    lum = np.zeros( (img.shape[0], img.shape[1]) )  # initilize 2d array with height, width
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel = img[row][col]
            lum[row][col] = pixel[0]*0.21 + pixel[1]*0.72 + pixel[2]*0.07

    return lum




if __name__ == "__main__":
    main()