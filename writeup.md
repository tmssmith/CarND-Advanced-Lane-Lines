## Advanced Lane Finding Project
### Thomas Smith

---

## Introduction

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The pipeline for a single images is:
* Apply a distortion correction to raw image
* Create a threshold image using color and gradient thresholds
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels using Sliding Windows technique
* Fit 2nd order polynimial to lane pixels
* Determine the curvature of the lane and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image
* Output final image with the lane boundaries and numerical estimation of lane curvature and vehicle position

The pipeline for a video is the same as for a single image with the following exceptions:
* Lanes found are stored for use when finding lanes in future frames
* If a good lane line was found in the previous frame then a Search From Prior technique is used instead of the Sliding Windows technique for finding lane pixels
* The final lane boundaries for a given frame are found as the weighted average of the lines form the current frame and 4 previous frames

[//]: # (Image References)

[image1]: ./output_images/undistort_chess.jpg "Undistorted calibrationimage"
[image2]: ./output_images/undistort_example.jpg "Undistorted road image"
[image3]: ./output_images/threshold.jpg "Binary Threshold Example"
[image4]: ./output_images/warp_pts.jpg "Warp source and destination points"
[image5]: ./output_images/warped.jpg "Warped image"
[image6]: ./output_images/sliding_window.jpg "Sliding windows"
[image7]: ./output_images/bestfit.jpg "Lines of best fit"
[image8]: ./output_images/lanelines.jpg "Final image"

[video1]: ./output_videos/project_video_output.mp4 "Video"

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

This section describes how the rubric points for this project have been met and provides examples.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Satisfied by this document

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This step is completed by the `cal_cam` function of the [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py) module, lines 503 to 530.

First we set up an empty array for the object points as  `obj_p`. Object points represent the coordinates in 3D space of the chessboard corners in the world.

The object points for a given calibration image are labelled `objp`. We assume all the chessboard corners lie on the same plane and so can define `objp` to be the same for each calibration image (as each image is of the same chessboard with the same number of corners. Hence `obj_p` contains one instance of `objp` for each calibration image.

Next we set up an empty array for the image points as `img_pts`. Image points represent the coordinates of the chessboard corners in 2D image space.

Chessboard corners are detected in each calibration image using the `cv2.findChessboardCorners` function and the coordinates of the corners are appended to 'img_pts'.

The `cv2.calibrateCamera` function is used to determine the camera calibration and distortion coefficients (`mtx` and `dst` respectively) and the `cv2.getOptimalNewCameraMatrix` function is used to find the new camera matrix (`newCameraMtx`) such that the undistorted images will not contain generated virtual (black) pixels in regions where the undistortion generates new data from the distorted image.

Using the `mtx`, `dist` and `newCameraMtx` parameters found above with the `cv2.undistort` function to apply distortion correction to one of the calibration images gives the below result:

![Undistorting a chessboard image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the `mtx`, `dist` and `newCameraMtx` parameters found above with the `cv2.undistort` function to apply distortion correction to one of the test images gives the below result:

![Undistorted image of road][image2]

This is carried out on line 36 of [`advanced_lane_finding.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/advanced_lane_finding.py)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A binary threshold image of the input image is generated using both color and gradient thresholds. This is carried out by the `threshold` function at lines 197 to 205 of [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py).

This function in turn calls the `gradientThreshold` and `colorThreshold` functions at lines 207 to 232 and lines 234 to 263 respectively of the same module.

An example threshold image is shown below:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The binary threshold image is warped to a birds-eye perspective using the `warpImage` function at lines 164 to 195 of the [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py) module.

This function warps the input image using source and destination points that are hardcoded at lines 31 to 38 in [`parameters.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/parameters.py). They are shown below for reference:

| Source        | Destination   |
|:-------------:|:-------------:|
| (590, 467)    | (320, 0)      |
| (721, 467)    | (1020, 0)     |
| (1020, 669)   | (1020, 669)   |
| (320, 669)    | (320, 669)    |

The `warpImage` function returns the warped image and the inverse transform matrix, `Minv`, used to unwarp the image back to the original undistorted perspective.

An undistorted image with the source and destination points is shown below.

![image with source and destination points][image4]

An image of a straight road section that has been warped to get a birds-eye view is shown below. The lane lines are still parallel, indicating the source and destination points used are reasonable.

![warped image][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Pixels that identify lane lines are detected using one of two techniques:
* Sliding Windows
* Search From Prior

The Sliding Windows technique doesn't rely on previous information but is more computationaly expensive.

The Search From Prior technique uses lines derived from previous images as a starting point and is computationally cheaper.

The `getLanePixels` function at lines 52 to 83 of [`advanced_lane_finding.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/advanced_lane_finding.py) uses a histogram method to identify the x positions where the most non-zero pixels are founds in either half (left and right) of the image. This is used as the starting point for the Sliding Windows technique.

This function also includes logic to decide which lane line pixel identification technique to use based on information about the lane lines found in previous frames.

The `sliding_windows` function at lines 366 to 396 of [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py) iterates up through the images height, placing a window and identifying non-zero pixels with the window. If enough non-zero pixels are located (above `minpix`) then the starting x point of the next window is updated to be the mean of the x co-ordinates of the identified non-zero pixels.  An image showing this technique is shown below. Non-zero pixels within the windows have been identified and color red and green for the left and right lines respectively.

![sliding windows][image6]

The `search_from_prior` function at lines 398 to 427 of [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py) implements the Search From Prior technique to identify non-zero pixels in the image in a region either side of the lane line identified in the previous image.

Once lane line pixels have been identified using either of the two techniques above, a 2nd order polynomial is fitted to the pixels for the left and right lanes using the numpy `np.polyfit` function.

The lines of best fit found for the image above are shown in the image below:
![lines of best fit][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


If a 2nd order polynomial is given by $x = Ay^{2}+By+C$ then the radius of curvature of a line is given by:

$$ R_{curve} = \frac{\left(1+\left(2Ay+B\right)^{2}\right)^{\frac{3}{2}}}{\left|2A\right|} $$

As the coefficients of the best fit polynomials found in this pipeline are in units of pixels, we must first convert them to units of pixels. Converted coefficients are given by:

$$ A_m = \frac{mx}{my^{2}}A_{px} $$
$$ B_m = \frac{mx}{my}B_{px} $$

where $A_m$, $B_m$ and $A_{px}$, $B_{px}$ are coefficients in meters and pixels respectively, and $mx$, $my$ are conversion factors expressed as meters per pixel. For this project:
$$mx = \frac{3.7}{700}$$
$$my = \frac{30}{720}$$

This conversion is carried out in the `convert_coeffs` function at lines 493 to 501 of [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py).


With the polynomial coefficients expressed in units of meters, the lane curvature is found at line 446 of [`lane_find_utils.py`](https://github.com/tmssmith/CarND-Advanced-Lane-Lines/blob/master/lane_find_utils.py).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

An example output image after this pipeline is completed is given below.

![Image with lane overlay][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is the result when the pipeline is applied to `project_video.mp4`

![Video with lane lines][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline developed in this project is likely to struggle in the following situations:

* Difficult lighting conditions such as night or heavy shadow
* Very bright conditions that lead to hazing of the camera view or lense flare
* Objects or vehicles blocking the view of the road ahead
* As the source and destination points for the persepctive transform are hardcoded the pipeline is vulnerable to changes in camera position relative to the vehicle and road surface, such as might be caused by bumps in the road, a heavy load changing the vehicles attitude, or a change to the camera's mounting location
* The pipeline developed in this project does not run in real time on the development hardware. A pipeline for an autonomous vehicle needs to run in realtime.

Potential approachs to improve this pipeline could include:
* Dynamic parameterisation
    * Thresholding parameters are currently hardcoded and must cater for many different image conditions. An ability to dynamically adjust these parameters based on previous frames could improve results
* Dynamic perspective transform
    * Using features identified in the image, such as an horizon, or lane lines from previous images, it could be possible to adjust the source and destination points for the perspective transform to observations
