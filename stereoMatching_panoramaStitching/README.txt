Computer Vision Project 2:
Simple Stereo Matching and Panorama Stitching w/ Homographies
README

Jiaming Zhao

Results from running the program.

PART I:

RMS gaussian w/ sigma=1:  11.613480804836552
RMS bilateral w/ d=4, sigma=1:  11.595132577098534
RMS consistency using bilateral:  8.1287588417321


PART II:

Best homography (differs slightly each run due to usage of RANSAC):

[[  7.31623763e-01  -1.94952364e-02   3.64836057e+02]
 [ -9.33985208e-02   9.43879266e-01   3.69041550e+01]
 [ -4.43032738e-04  -1.76283036e-05   1.00000000e+00]]

number of inliers: 245
out of total number of matches: 253


How to run program with from the command line:

--------------------------------------------------------------- Install Packages
Ensure that Numpy is installed already

Download OpenCV from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv 
	opencv_python-3.2.0-cp36-cp36m-win_amd64.whl 
	opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl 
(Need both opencv and contrib packages) 

Install OpenCV using command line 
	pip install opencv_python-3.2.0-cp36-cp36m-win_amd64.whl 
	pip install opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl


--------------------------------------------------------------- Run Program

Open proj_02.py as text and check that the routines - main_stereo() and main_panorama()
are not commented out. Choose which routine to run and uncomment that one.

Use command line
	python proj_02.py

Stereo Matching portion displays and prints:
	Gaussian disparity map image
	Bilateral disparity map image
	Bilateral Right disparity map image
	Left-Right Consistency Check disparity map image

	RMS values for Gaussian, Bilateral, and Consistency

Panorama portion displays and prints:
	Number of inliers
	Number of total matches
	The best homography found

	Panorama output image
		- option to save panorama output image into src folder
		- press 's' key to save when image opens in window

--------------------------------------------------------------- End

