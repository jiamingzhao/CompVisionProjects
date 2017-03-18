'''
    Jiaming Zhao
    March 2017
    Project 2: Simple Stereo Matching and Panorama Stitching w/ Homographies
    Computer Vision @ University of Virginia
'''

# -*- coding: utf-8 -*-
import skimage, skimage.io, skimage.transform, pylab, skimage.filters, cv2, math
import numpy as np


def main():
    # Simple Stereo Matching using resized images from Middlebury Stereo dataset
    main_stereo()

    # Panorama Stitching using Homographies and the RANSAC method
    main_panorama()


def main_panorama():
    img_a = cv2.imread('pair_a_small.jpg')
    img_b = cv2.imread('pair_b_small.jpg')

    # SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img_a, None)
    kp2, des2 = sift.detectAndCompute(img_b, None)

    # use Brute-Force Matcher to test all the descriptors for matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # reject failures of ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    # use RANSAC to compute the best homography that maps B coordinates to A coordinates
    best_homography = ransac_estimation(good_matches, kp1, kp2)
    print(best_homography)

    # warp image B to image A's coodinate system using the best homography found in RANSAC
    # use helper routine provided by @ Professor Connelly Barnes at University of Virginia
    final_img = composite_warped(img_a, img_b, best_homography)

    cv2.imshow('Panorama', final_img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('panorama.png',final_img)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()


def main_stereo():
    # load the left and right camera images and convert to float arrays
    img_left = skimage.io.imread('./im0_small.png')  # left image
    img_right = skimage.io.imread('./im1_small.png')  # right image
    img_left = skimage.img_as_float(img_left)       # convert image to float
    img_right = skimage.img_as_float(img_right)       # convert image to float

    # establish max disparity
    h, w = len(img_left), len(img_left[0])
    max_disparity = int(h / 3)  # max disparity amount set to height/3

    # find the disparity space image (DSI) taking the left image as the base
    dsi_left = get_DSI_left(img_left, img_right, h, w, max_disparity)

    # perform spatial aggregation using gaussian filter and bilateral filter
    dsi_g = dsi_gaussian(dsi_left, max_disparity, 1)
    dsi_b = dsi_bilateral(dsi_left, img_left, max_disparity)

    # find the disparity for each coordinate based on the smallest dsi value (using argmin)
    disp_map_g = smallest_DSI(dsi_g, h, w)
    print('Gaussian disparity map')
    pylab.imshow(disp_map_g)
    pylab.show()

    disp_map_b = smallest_DSI(dsi_b, h, w)
    print('Bilateral disparity map')
    pylab.imshow(disp_map_b)
    pylab.show()

    # test the difference between the results of this program and the "ground truth values" of a file
    rms_gaussian = comp_disp_truth(disp_map_g, h, w)
    rms_bilateral = comp_disp_truth(disp_map_b, h, w)
    print('RMS gaussian w/ sigma=1: ', rms_gaussian)
    print('RMS bilateral w/ d=4, sigma=1: ', rms_bilateral)


    # begin left-right consistency check
    # get dsi_right and perform stereo matching on it
    dsi_right = get_DSI_right(img_left, img_right, h, w, max_disparity)
    dsi_b_r = dsi_bilateral(dsi_right, img_left, max_disparity)
    disp_map_right_b = smallest_DSI(dsi_b_r, h, w)

    print('Bilateral Right disparity map')
    pylab.imshow(disp_map_right_b)
    pylab.show()

    # check if corresponding disparities differ by more than a particular threshold (threshold=15)
    # to find occlusions (mismatched regions that appear in one image and not the other)
    occ = find_occlusions(disp_map_b, disp_map_right_b, h, w)
    print('Left-Right Consistency Check disparity map')
    pylab.imshow(disp_map_b)
    pylab.show()

    # test the difference between the results of the consistency check minus occlusions
    # less difference / more accurate
    rms_consistency = comp_disp_truth(disp_map_b, h, w, occ)
    print('RMS consistency using bilateral: ', rms_consistency)


def composite_warped(a, b, H):  # routine provided by @ Professor Connelly Barnes
    # "Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a coordinates."
    out_shape = (a.shape[0], 2*a.shape[1])                               # Output image (height, width)
    p = skimage.transform.ProjectiveTransform(np.linalg.inv(H))       # Inverse of homography (used for inverse warping)
    bwarp = skimage.transform.warp(b, p, output_shape=out_shape)         # Inverse warp b to a coords
    bvalid = np.zeros(b.shape, 'uint8')                               # Establish a region of interior pixels in b
    bvalid[1:-1,1:-1,:] = 255
    bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)    # Inverse warp interior pixel region to a coords
    apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape))) # Pad a with black pixels on the right
    return skimage.img_as_ubyte(np.where(bmask==1.0, bwarp, apad))    # Select either bwarp or apad based on mask

def ransac_estimation(matches, kp1, kp2):
    from random import sample
    max_iter = 1000  # most times loop will search for best match and inliers

    # calculate inliers and test homographies
    largest_inlier_count = 0
    best_homog = None

    for _ in range(max_iter):
        # find random samples from all the matches to begin RANSAC
        samp = sample(matches, 4)

        # extract the 4 sample matches as points in list
        pa_list = []
        pb_list = []
        for m in samp:
            img1_idx = m[0].queryIdx
            img2_idx = m[0].trainIdx

            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            pa_list.append( (x1, y1) )
            pb_list.append( (x2, y2) )

        # find the homography from the random sample
        homog = get_homography(pa_list, pb_list)

        # extract matches as points in list for all matches to test the homography from sample
        matA_list = []
        matB_list = []
        for mat in matches:
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            matA_list.append( (x1, y1) )
            matB_list.append( (x2, y2) )

        # apply homography to point b, and find distance between Hb and point a.
        # if difference/distance between Hb and point a is below threshold, it is an inlier (to ignore noise)
        inliers = 0
        for v in range(len(matB_list)):
            xy1 = [ matB_list[v][0], matB_list[v][1], 1 ]
            hbxy = apply_homography( homog, xy1 )  # calculate Hb

            # calculate sum of squared differences between Hb values and a values
            x_diff = hbxy[0] - matA_list[v][0]
            y_diff = hbxy[1] - matA_list[v][1]
            dist = math.sqrt( (x_diff)**2  + (y_diff)**2 )

            # compare simple distance between hbxy and ax, ay
            if dist < 5:  # threshold held at 5 distance between point Hb and point a
                inliers += 1

        # update largest_inlier_count and best homography
        if inliers > largest_inlier_count:
            largest_inlier_count = inliers
            best_homog = homog

    print('Number of inliers: ', largest_inlier_count)
    print('out of Number of total matches: ', len(matB_list))
    return best_homog

def get_homography(pa, pb):  # Ax=b solves for x (homography)
    # fitting a homography by taking a list of four points in img_a and four points in img_b
    # use 8x8 matrix multiplied by 8x1 array to find the 8 values needed to fill the 3x3 homography
    # https://inst.eecs.berkeley.edu/~cs194-26/fa14/upload/files/proj7B/cs194-fb/images/homography.png
    # A = 8x8 matrix, b = 8x1 array... numpy.linalg.lstsq(a, b) solves equation Ax=b for x, which is the homography

    A = np.zeros( (8, 8) )
    b = []

    for x in range(4):  # insert values to b array
        b.append(pa[x][0])
        b.append(pa[x][1])

    b = np.array(b)

    for i in range(0, 8, 2):
        pbx, pby = pb[ int(i/2) ]
        pax, pay = pa[ int(i/2) ]
        A[i] = [ pbx, pby, 1, 0, 0, 0, -pbx*pax, -pby*pax ]
        A[i+1] = [ 0, 0, 0, pbx, pby, 1, -pbx*pay, -pby*pay ]

    h = ( np.linalg.lstsq(A, b) )[0]
    homog = [ [h[0], h[1], h[2]],
              [h[3], h[4], h[5]],
              [h[6], h[7], 1]
              ]
    return np.matrix(homog)

def apply_homography(hg, b_point):
    denom = np.sum( np.multiply(hg[2], b_point) )
    numerator_x = np.sum( np.multiply(hg[0], b_point) )
    numerator_y = np.sum( np.multiply(hg[1], b_point) )

    a_x = round(numerator_x/denom)
    a_y = round(numerator_y/denom)
    return a_x, a_y

def find_occlusions(disp_left, disp_right, h, w):  # checks consistency between left and right and saves into left image
    occlusions = np.zeros( (h, w) )
    for i in range(h):
        for j in range(w):
            offset = int( disp_left[i][j] )
            if abs( disp_left[i][j] -  disp_right[i][j-offset] ) > 15:
                disp_left[i][j] = 0
                occlusions[i][j] = 1  # mark occlusions that do not show up left vs right
    return occlusions

def dsi_bilateral(dsi, left, max_disparity):
    slices_list = []
    for m in range(max_disparity):
        # take 2d slice of 3d DSI and luminance of left
        dsi_slice = dsi[:, :, m]
        left_gray = luminance(left)

        # convert to float32 arrays
        dsi_slice32 = dsi_slice.astype(np.float32)
        left_gray = left_gray.astype(np.float32)

        # NOTE: bilateralFilter works just as well as jointBilateralFilter
        # perform the bilateral filter on all slices and add to the stack of 2d images
        # dsi_slice = cv2.bilateralFilter(dsi_slice32, 4, 1, 1)
        dsi_slice = cv2.ximgproc.jointBilateralFilter(left_gray, dsi_slice32, 4, 1, 1)
        slices_list.append(dsi_slice)
    return np.dstack(slices_list)

def dsi_gaussian(dsi, max_disparity, sig):
    slices_list = []
    for m in range(max_disparity):
        dsi_slice = dsi[:, :, m]
        dsi_slice = skimage.filters.gaussian(dsi_slice, sigma=sig)
        slices_list.append(dsi_slice)
    return np.dstack(slices_list)

def comp_disp_truth(disp_map, h, w, occlusions=None):  # comparing disparity map with ground truth provided
    num_elements = h*w
    g_truth = np.load('./gt.npy')
    tot = 0

    if occlusions is not None:
        for i in range(h):
            for j in range(w):  # calculate the difference and square each
                if occlusions[i][j] == 1:
                    num_elements -= 1
                else:
                    tot += ( disp_map[i][j] - g_truth[i][j] ) ** 2
    else:
        for i in range(h):
            for j in range(w):  # calculate the difference and square each
                tot += ( disp_map[i][j] - g_truth[i][j] ) ** 2

    return math.sqrt( tot / num_elements )  # sqrt of the mean

def smallest_DSI(dsi, h, w):  # generating disparity map
    disp_map = np.zeros( (h, w) )
    for i in range(h):
        for j in range(w):
            disp_map[i][j] = np.argmin( dsi[i][j] )
    return disp_map

def get_DSI_left(left, right, h, w, max_disparity):  # find the disparity space image (DSI) as 3D array (x, y, d)
    dsi = np.zeros( (h, w, max_disparity) )  # 3d array with x, y, and 1 disparity value

    for i in range(h):
        for j in range(w):
            for k in range(max_disparity):
                if (j-k < 0):
                    tot = 3
                else:
                    lefty = left[i][j]  # left RGB values
                    righty = right[i][j-k]  # right RGB values
                    tot = 0
                    for color in range(3):  # sum of squares of each rgb color
                        tot += ( (lefty[color] - righty[color]) ** 2 )
                dsi[i][j][k] = tot
    return dsi

def get_DSI_right(left, right, h, w, max_disparity):
    dsi = np.zeros( (h, w, max_disparity) )  # 3d array with x, y, and 1 disparity value

    for i in range(h):
        for j in range(w):
            for k in range(max_disparity):
                if (j+k >= w):
                    tot = 3
                else:
                    lefty = left[i][j+k]
                    righty = right[i][j]
                    tot = 0
                    for color in range(3):
                        tot += ( (lefty[color] - righty[color]) ** 2 )
                dsi[i][j][k] = tot
    return dsi

def luminance(img):
    lum = np.zeros( (img.shape[0], img.shape[1]) )  # initilize 2d array with height, width

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel = img[row][col]
            lum[row][col] = pixel[0]*0.21 + pixel[1]*0.72 + pixel[2]*0.07

    return lum


if __name__ == "__main__":
    main()
