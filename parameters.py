"""
Parameters for lane finding with the project_video.mp4
"""

### LINE SMOOTHING ###
# Number of previous lane lines used for
# finding weighted average of lane lines
WGTD_MVNG_AVG_LEN = 5


### CAMERA CALIBRATION ###
# Number of chessboard corners in x and y
NX = 9
NY = 6

### PIXEL TO METER CONVERSION ###
# Conversion from pixels to real world
# measurements (pixels to meters)
MX = 3.7/700
MY = 30/720

### PX COUNT REQD TO DEFINE A LINE ###
# Minimum number of pixels that must be
# detected for a new line to be derived
MINPIX_LINE = 1000

### WARP IMAGE PERSPECTIVE ###
# Points used to warp the perspective
# of the image to a birds-eye view

# src pts
P1_SRC = (590, 467)
P2_SRC = (721, 467)
P3_SRC = (1020, 669)
P4_SRC = (320, 669)

# dst pts
DST_Y = 0

### GRADIENT THRESHOLDS ###
# Kernels size and threshold values used
# for producing absolute gradient,
# gradient magnitude and gradient
# direction thresholds of images
KERNEL_ABS_X = 9
KERNEL_ABS_Y = 7
KERNEL_MAG = 7
KERNEL_DIR = 23

THRESH_ABS_X = (25,180)
THRESH_ABS_Y = (25,180)
THRESH_MAG = (100,255)
THRESH_DIR = (0.7,1.3)

### COLOR THREHOLDS ###
# Threshold values used to generate color
# thresholds of images
THRESH_R = (210, 255)
THRESH_H = (20, 50)
THRESH_S = (120, 200)

### SLIDING WINDOWS ###
# Parameters of sliding windows used to
# determine pixels that correspond to
# lane lines
W_WIDTH = 200       # window width
W_HEIGHT = 100      # window height
# minimum number of pixels detected before
# adjusting x-position of new window
MINPIX_WINDOW = 50

### SEARCH FROM PRIOR ###
# Parameters of area used to determine
# pixels that correspond to lane lines
# when search from prior is used

# Minimum number of px in previous line before search from prior is used
MINPIX_PRIOR = 5000
# Width of search from prior area either side of previous line
MARGIN = 75
