# -*- coding: utf-8 -*-
import skimage, skimage.io, pylab, scipy.ndimage.filters
import numpy as np


def main():
    orig_img = skimage.io.imread('./flower_in.jpg')  # read image
    orig_img = skimage.img_as_float(orig_img)       # convert image to float
    gr_img = luminance(orig_img)                  # call luminance subroutine for 2d array

    # convolve with gaussian and get derivatives through sobel filters
    conv_g_img = scipy.ndimage.filters.convolve(gr_img, get_gaussian_filter())
    conv_sx_img = scipy.ndimage.filters.convolve(conv_g_img, get_sobel(1))
    conv_sy_img = scipy.ndimage.filters.convolve(conv_g_img, get_sobel(0))

    # perform canny edge detection
    canny_edge(conv_sx_img, conv_sy_img)

    # perform harris corner detection
    harris_corner(conv_sx_img, conv_sy_img, orig_img)


def harris_corner(conv_sx_img, conv_sy_img, orig_img):
    corners_list = []
    m = 4
    eig_thresh = 500000

    for i in range(m, len(conv_sx_img)-m):
        for j in range(m, len(conv_sx_img[0])-m):
            # take the smaller window
            harr_arr = np.zeros( (2, 2) )
            for x in range(-m-1, m):
                for y in range(-m-1, m):
                    fx = conv_sx_img[i+x][j+y]
                    fy = conv_sy_img[i+x][j+y]

                    harr_arr[0][0] = fx**2
                    harr_arr[0][1] = fx*fy
                    harr_arr[1][0] = fx*fy
                    harr_arr[1][1] = fy**2

            eig = np.linalg.det(harr_arr) - ( 0.04*np.trace(harr_arr) ) ** 2
            if abs(eig) > eig_thresh:
                corners_list.append( (eig, i, j) )

    # maximum suppression
    corners_list.sort()
    corners_list_temp = corners_list[:]
    num_pops = 0

    for c1 in range(len(corners_list_temp)):
        x1 = corners_list_temp[c1][1]
        y1 = corners_list_temp[c1][2]

        for c2 in range(c1, len(corners_list_temp)):
            x2 = corners_list_temp[c2][1]
            y2 = corners_list_temp[c2][2]

            if x1-10 <= x2 <= x1+10 and y1-10 <= y2 <= y1+10 and x1 != x2:
                corners_list.pop(c1-num_pops)
                num_pops += 1
                break

    num_corners = 80 if len(corners_list) > 80 else len(corners_list)
    for u in range(5, num_corners-5):
        for v in range(-5, 5):
            orig_img[ corners_list[u][1] ][ corners_list[u][2] + v ] = [0, 15, 135]
            orig_img[ corners_list[u][1] + v ][ corners_list[u][2] ] = [0, 15, 135]


    pylab.imshow(orig_img)
    pylab.show()


def canny_edge(conv_sx_img, conv_sy_img):
    # compute the edge strength
    edge_strength = np.sqrt( np.add( np.square(conv_sx_img), np.square(conv_sy_img) ) )
    # compute the edge orientation D = atan(Fy/Fx)
    edge_orient = np.arctan( np.divide(conv_sy_img, conv_sx_img) )

    # find nearest direction in orientation then thin image using non-maximum suppression
    round_directions(edge_orient)

    thinned_edge_img = get_thinned_edge(edge_strength, edge_orient)
    # part 1 of hysteresis thresholding
    low = 19
    high = 32
    hyst_arr = hyst_thresh(low, high, thinned_edge_img)

    # part 2 of hysteresis thresholding: go through every pixel and determine perform DFS on strong pixels
    for row in range(1, len(hyst_arr)-1):
        for col in range(1, len(hyst_arr[0])-1):
            val = hyst_arr[row][col]
            if val == 2:  # if strong pixel, look around for weak pixels
                flood_fill(row, col, thinned_edge_img, hyst_arr)

    pylab.imshow(thinned_edge_img, cmap="gray")
    pylab.show()

def flood_fill(x, y, img, hyst):
    if x < 0 or y < 0 or x >= len(hyst) or y >= len(hyst[0]):
        return
    if hyst[x][y] <= 0: 
        hyst[x][y] = -1
        return
    elif hyst[x][y] == 2:
        hyst[x][y] = -1
        img[x][y] = 255
        flood_fill(x-1, y-1, img, hyst)
        flood_fill(x-1, y, img, hyst)
        flood_fill(x-1, y+1, img, hyst)
        flood_fill(x+1, y-1, img, hyst)
        flood_fill(x+1, y, img, hyst)
        flood_fill(x+1, y+1, img, hyst)
        flood_fill(x, y-1, img, hyst)
        flood_fill(x, y+1, img, hyst)
    elif hyst[x][y] == 1:
        hyst[x][y] = 2
        flood_fill(x, y, img, hyst)

def hyst_thresh(t_low, t_high, img):
    hyst_arr = np.zeros( (len(img), len(img[0])) )
    for row in range(len(img)):
        for col in range(len(img[0])):
            compare_val = img[row][col]
            if compare_val < t_low:
                img[row][col] = 0
                hyst_arr[row][col] = 0
            elif compare_val > t_high:
                hyst_arr[row][col] = 2
            else:
                hyst_arr[row][col] = 1
    
    return hyst_arr

def get_thinned_edge(stren, orient):
    # thin image - follow edge_orient's direction. if curr edge_stren < dir edge_stren: set TEI to edge_stren val
    thin_edge = np.zeros( (len(stren), len(stren[0])) ).astype(float)

    for row in range(1, len(stren)-1):
        for col in range(1, len(stren[0])-1):
            stren_val = stren[row][col]
            orient_val = orient[row][col]
            if orient_val == 0 and ( stren_val < stren[row][col+1] or stren_val < stren[row][col-1] ):
                thin_edge[row][col] = 0
            elif orient_val == 1 and ( stren_val < stren[row-1][col+1] or stren_val < stren[row+1][col-1] ):
                thin_edge[row][col] = 0
            elif orient_val == 2 and ( stren_val < stren[row-1][col] or stren_val < stren[row+1][col] ):
                thin_edge[row][col] = 0
            elif orient_val == -1 and ( stren_val < stren[row+1][col+1] or stren_val < stren[row-1][col-1] ):
                thin_edge[row][col] = 0
            else:
                thin_edge[row][col] = stren[row][col]
    
    return thin_edge

def round_directions(orient):  # find shortest distance between each orientation number and [0, pi/4, pi/2, 3pi/4]
    valid_directions = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    actual_directions = [2, -1, 0, 1, 2]
    
    for row in range(len(orient)):
        for col in range(len(orient[0])):  # valid_directions[ index of min(number-valid_possibilities)^2 ]
            orient[row][col] = actual_directions[ np.argmin( (orient[row][col] - valid_directions) ** 2) ]

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
