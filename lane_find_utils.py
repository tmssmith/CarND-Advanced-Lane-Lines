"""
A set of utility classes and functions used for advanced lane finding
"""

import os
import fnmatch
import numpy as np
import cv2
from parameters import *


class Window:
    """
    Rectangular window defining an image region for identifying non-zero pixels

    Window dimensions are constrained to ensure the window does not exceed the
    dimensions of the input image.

    Args:
        img (int 2D array): target image for detecting non-zero pixels
        centre_x (int): centre of the window on the x-axis
        base_y (int): bottom of the widow on the y-axis
        width (int): width of the window
        height (int): height of the window

    Attributes:
        img (int 2D array): target image for detecting non-zero pixels
        centre_x (int): centre of the window on the x-axis
        p1 (int tuple): Top-left vertex of window
        p2 (int tuple): Top-right vertex of window
        p3 (int tuple): Bottom-right vertex of window
        p4 (int tuple): Bottom-left vertex of window
        pts (int 2D array): Window vertices
        window (int 2D array): Pixel values of `img` in window region
        img_inds (int 2D array): `img` indices of non-zero pixels in window
    """

    def __init__(self, img, centre_x, base_y, width, height):
        """
        Define window vertices, identify encompassed pixels from source image
        """
        self.img = img
        self.centre_x = int(centre_x)
        self.p1 = (int(max(0,centre_x - width//2)),int(max(0,base_y - height)))
        self.p2 = (int(centre_x + width//2), int(base_y - height))
        self.p3 = (int(min(img.shape[1], centre_x + width//2)), int(base_y))
        self.p4 = (int(centre_x - width//2), int(base_y))
        self.pts = np.array([self.p1,self.p2,self.p3,self.p4], dtype=np.int32)
        self.window = np.array(img[self.p1[1]:self.p3[1], self.p1[0]:self.p3[0]])

    def draw_window(self, img, color = (200,200,0)):
        """Draw a rectange on img representing window"""
        cv2.rectangle(img, self.p1, self.p3, color, 2)

    def get_pixels(self):
        """
        Return indices of non-zero pixels from window region of input image

        Returned indices are in the input image frame of reference
        """
        window = self.window
        # window_inds gives positions of nonzero pixels in window frame of reference
        # img_inds gives positions of nonzero pixels in image frame of reference
        window_inds = np.transpose(window.nonzero())
        # Need to swap window_inds columns around as openCV has coordinates as (x,y) but pixels as (row, column)...
        window_inds[:,[1,0]] = window_inds[:,[0,1]]
        self.img_inds = np.add(window_inds, self.p1)

        return self.img_inds

    def draw_pixels(self, img, color = (0,0,255)):
        """
        Set color of non-zero pixels in window

        Args:
            img (int 2D array): Destination image to set pixel colors
            color: (int tuple, optional): BGR color to set pixels to
        Attributes:
            img_inds (int 2D array): `img` indices of non-zero pixels in window
        """
        img_inds = self.img_inds
        img[img_inds[:,1], img_inds[:,0]] = color

    def get_next_centrex(self):
        """Returns mean x position of non-zero pixels within window"""
        if len(self.img_inds) >= MINPIX_WINDOW:
            new_centre_x = np.int(np.mean(self.img_inds[:,0]))
            return new_centre_x
        else:
            return self.centre_x


class Line:
    """
    Best fit line (2nd order) for a given set of pixels and previous lines

    Args:
        px (int 2D array): indices of pixels that define line
        linetracker (list): List of lines from previous frames

    Attributes:
        px (int 2D array): indices of pixels that define line
        px_cnt (int): Number of pixels in px
        coeffs (float 1D array): Coefficients of 2nd order line of best fit
    """

    def __init__(self, px, linetracker):
        self.px = px
        if self.px is not None:
            self.px_cnt = self.px.shape[0]
        else:
            self.px_cnt = 0
        self.coeffs = self.fit_line(linetracker)

    def fit_line(self, linetracker):
        """
        Returns coefficients of 2nd order line of best fit

        Find coefficients of 2nd order line of best fit based an input set of
        pixels. If the quantity of pixels in the input set is not above
        MINPIX_LINE then the coefficients of the previous line are used.
        If no previous line is available then the coefficients are set to `None`
        """
        if self.px_cnt >= MINPIX_LINE:
            self.coeffs = np.polyfit(self.px[:,1], self.px[:,0], 2)
        elif not linetracker:
            self.coeffs = np.array([None, None, None])
        else:
            self.coeffs = linetracker[-1].coeffs

        return self.coeffs


def bgr2rgb(img):
    """Returns RGB representation of BGR input image"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb

def bgr2gray(img):
    """Returns grayscale representation of BGR input image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def rgb2gray(img):
    """Returns grayscale representation of RGB input image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray

def gray2bgr(img):
    """Returns BGR representation of grayscale input image"""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return bgr

def bgr2hls(img):
    """Returns HLS representation of BGR input image"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    return hls

def warpImage(img):
    """
    Returns a warped view of input image and the inverse distortion matrix

    Source points are defined in `parameters.py` and are defined to generate a
    birds-eye view of the input image

    Args:
        img (int 2D array): array representation of input image

    Returns:
        img_warp (int 2D array): array representation of warped input image
        Minv (2D array): inverse transformation matrix back to original view
    """
    p1_src = P1_SRC             # Top-left source point
    p2_src = P2_SRC             # Top-right source point
    p3_src = P3_SRC             # Bottom-right source point
    p4_src = P4_SRC             # Bottom-left source point
    dst_y = DST_Y               # Height of top destination points
    p1_dst = [p4_src[0],dst_y]  # Top-left destination point
    p2_dst = [p3_src[0],dst_y]  # Top-right destination point
    p3_dst = p3_src             # Bottom-right destination point
    p4_dst = p4_src             # Bottom-left destination point
    src = np.array([p1_src, p2_src, p3_src, p4_src], dtype = np.int32)
    dst = np.array([p1_dst, p2_dst, p3_dst, p4_dst], dtype=np.int32)

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    img_warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return img_warp, Minv

def threshold(img):
    """
    Returns the binary threshold image using gradient and color thresholds
    """
    grad_thresh = gradientThreshold(img)
    color_thresh = colorThreshold(img)
    img_threshold = grad_thresh | color_thresh

    return img_threshold

def gradientThreshold(img):
    """
    Returns the binary gradient threshold of the input image

    Uses Sobel filters to generate image binaries for absolute Sobel gradients
    in x and y, Sobel gradient magnitudes and Sobel gradient directions. Final
    binary output is:
        (absolute_x & absolute_y) | (magnitude & direction)

    Sobel filter kernel sizes and thersholds are defined in `parameters.py`

    Args:
        img (int 2D array): array representation of input image

    Returns:
        grad_binary (int 2D array): gradient threshold of `img`
    """
    img_gray = bgr2gray(img)
    sobelx_binary = sobel_abs_thresh(img_gray, 'x', KERNEL_ABS_X, THRESH_ABS_X)
    sobely_binary = sobel_abs_thresh(img_gray, 'y', KERNEL_ABS_Y, THRESH_ABS_Y)
    sobel_mag_binary = sobel_mag_thresh(img_gray, KERNEL_MAG, THRESH_MAG)
    sobel_dir_binary = sobel_dir_thresh(img_gray, KERNEL_DIR, THRESH_DIR)
    grad_binary = \
        (sobelx_binary & sobely_binary) | (sobel_mag_binary & sobel_dir_binary)

    return grad_binary

def colorThreshold(img):
    """
    Returns the binary color threshold of the input image

    Color threshold binaries are generated for the R channel of the RGB image
    colorspace and the H and S channels of the HLS image colorspace. Final
    binary output is:
        R_binary | (H_binary & S_binary)

    Color thersholds are defined in `parameters.py`

    Args:
        img (int 2D array): array representation of input image

    Returns:
        color_binary (int 2D array): color threshold of `img`
    """
    img_rgb = bgr2rgb(img)
    img_hls = bgr2hls(img)
    r = img_rgb[:,:,0]
    h = img_hls[:,:,0]
    s = img_hls[:,:,2]

    r_binary = cv2.inRange(r, THRESH_R[0], THRESH_R[1])
    h_binary = cv2.inRange(h, THRESH_H[0], THRESH_H[1])
    s_binary = cv2.inRange(s, THRESH_S[0], THRESH_S[1])

    color_binary = r_binary | (h_binary & s_binary)

    return color_binary

def sobel_abs_thresh(img, orient='x', k_size=3, threshold=(0,255)):
    """
    Returns a binary mask of pixel gradients in x or y using a Sobel filter

    Args:
        img (int 2D array): array representation of input image
        orient (str, optional): Orientation on which to apply the Sobel filter.
                                Accepted values are 'x' or 'y'. Defaults to 'x'
        k_size (int, optional): Kernel size of Sobel filter
                                Expects an odd number
                                Defaults to 3
        threshold (int tuple): Threshold values applied to the output of the
                               Sobel filter
                               Expects threshold to be of the form (lo, hi)
                               Defaults to (0, 255)

    Returns:
        mask (int 2D array): binary mask of pixel gradients
    """
    sobel = get_sobel(img, orient, k_size)
    abs_sobel = np.absolute(sobel)
    thresh_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mask = cv2.inRange(thresh_sobel, threshold[0], threshold[1])

    return mask

def sobel_mag_thresh(img, k_size=3, threshold=(0,255)):
    """
    Returns a binary mask of pixel gradient magnitudes using Sobel filters

    Args:
        img (int 2D array): array representation of input image
        k_size (int, optional): Kernel size of Sobel filter
                                Expects an odd number
                                Defaults to 3
        threshold (int tuple): Threshold values applied to the output of the
                               Sobel filter
                               Expects threshold to be of the form (lo, hi)
                               Defaults to (0, 255)

    Returns:
        mask (int 2D array): binary mask of pixel gradient magnitudes
    """
    sobelx = get_sobel(img, 'x', k_size)
    sobely = get_sobel(img, 'y', k_size)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag_abs = np.uint8(255*sobel_mag/np.max(sobel_mag))
    mask = cv2.inRange(sobel_mag_abs, threshold[0], threshold[1])

    return mask

def sobel_dir_thresh(img, k_size=3, threshold=(0,np.pi/2)):
    """
    Returns a binary mask of pixel gradient directions using Sobel filters

    Args:
        img (int 2D array): array representation of input image
        k_size (int, optional): Kernel size of Sobel filter
                                Expects an odd number
                                Defaults to 3
        threshold (int tuple): Threshold values applied to the output of the
                               Sobel filter
                               Expects threshold to be of the form (lo, hi)
                               Defaults to (0, pi/2)

    Returns:
        mask (int 2D array): binary mask of pixel gradient directions
    """
    sobelx = get_sobel(img, 'x', k_size)
    sobely = get_sobel(img, 'y', k_size)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    mask = cv2.inRange(sobel_dir, threshold[0], threshold[1])

    return mask

def get_sobel(img, orient='x', k_size=3):
    """
    Returns the output of a Sobel filter applied to the input image

    Args:
        img (int 2D array): array representation of input image
        orient (str, optional): Orientation on which to apply the Sobel filter.
                                Accepted values are 'x' or 'y'. Defaults to 'x'
        k_size (int, optional): Kernel size of Sobel filter
                                Expects an odd number
                                Defaults to 3

    Returns:
        sobel (int 2D array): output of Sobel filter
    """
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k_size)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = k_size)
    else:
        print('Orientation must be "x" or "y"')

    return sobel

def sliding_windows(img, centre_x):
    """
    Returns pixels that have been detected as lane line using sliding windows

    Iteratively creates Windows on `img` and detects non-zero pixels inside the
    window. Returns the X,Y values for the detected non-zero pixels.

    Args:
        img (int 2D array): array representation of input image
        centre_x (int): centre of first window on the x-axis

    Returns:
        lane_pts (int 2D array): Pixels identified as representing lane line
    """
    w_width = W_WIDTH
    w_height = W_HEIGHT
    img_height = img.shape[0]
    img_width = img.shape[1]
    lane_pts = []
    lane_color = (0,0,255)
    base_y = img_height

    for win in range(img_height // w_height + (img_height % w_height > 0)):
        window = Window(img, centre_x, base_y, w_width, w_height)
        px_inds = window.get_pixels()
        lane_pts.append(px_inds)
        centre_x = window.get_next_centrex()
        base_y -= w_height
    lane_pts = np.concatenate(lane_pts)

    return lane_pts

def search_from_prior(img, line):
    """
    Returns pixels that have been detected as lane line using previous lines

    Creates an area on `img` based on a previous line and a margin. Detects
    non-zero pixels inside the area. Returns the X,Y values for the detected
    non-zero pixels.

    Args:
        img (int 2D array): array representation of input image
        line (Line object): line produced from previous frame

    Returns:
        lane_pts (int 2D array): Pixels identified as representing lane line
    """
    margin = MARGIN
    lane_pts = []
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    coeffs = line.coeffs
    prev_line_x = np.polyval(coeffs, nonzeroy)

    inds = ((nonzerox >= prev_line_x - margin) &
        (nonzerox <= prev_line_x + margin)).nonzero()[0]
    x = nonzerox[inds]
    y = nonzeroy[inds]
    lane_pts = np.transpose(np.vstack((x,y)))

    return lane_pts

def draw_lane(img, leftlines, rightlines, Minv):
    """
    Returns `img` with lane line overlay, curvature and lane position info
    """
    mx = MX
    my = MY
    left_coeffs_px = wgtd_avg(leftlines)
    right_coeffs_px = wgtd_avg(rightlines)
    left_coeffs_m = convert_coeffs(left_coeffs_px, mx, my)
    right_coeffs_m = convert_coeffs(right_coeffs_px, mx, my)

    mid_coeffs_m = (left_coeffs_m + right_coeffs_m) / 2
    y_eval = img.shape[0]*my

    A = mid_coeffs_m[0]
    B = mid_coeffs_m[1]
    C = mid_coeffs_m[2]
    curv = (1+(2*A*y_eval + B)**2)**(3/2) / abs(2*A)

    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_x = np.polyval(left_coeffs_px, plot_y)
    right_x = np.polyval(right_coeffs_px, plot_y)
    left_pts = np.transpose(np.vstack((left_x, plot_y)))
    right_pts = np.transpose(np.vstack((right_x, plot_y)))

    left_lane_pos = np.polyval(left_coeffs_m, y_eval)
    right_lane_pos = np.polyval(right_coeffs_m, y_eval)
    lanecentre = ((left_lane_pos + right_lane_pos) / 2) * mx
    veh_center = img.shape[1]/2 * mx
    laneoffset = lanecentre - veh_center

    # Create an image to draw the lane guide on
    canvas = np.zeros_like(img)
    pts = np.vstack((left_pts, right_pts[::-1]))
    cv2.fillPoly(canvas, np.int32([pts]), (0,255, 0))

    # Use inverse perspective matrix (Minv) to warp canvas back to original image space
    lane_warp = cv2.warpPerspective(canvas, Minv, (img.shape[1], img.shape[0]))
    img_out = cv2.addWeighted(img, 1, lane_warp, 0.3, 0)

    cv2.line(img_out, (int(lanecentre / mx), img_out.shape[0]),(int(lanecentre / mx), img_out.shape[0] - 50), (0,0,0),3)
    cv2.line(img_out, (int(veh_center / mx), img_out.shape[0]),(int(veh_center / mx), img_out.shape[0]- 40),(255,0,0),3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, 'Lane curvature: {:.3f} km'.format(curv / 1000), (50,50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_out, 'Lane position: {:.2f} m'.format(laneoffset), (50,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_out

def wgtd_avg(lines):
    """
    Returns the weighted average of a set of line coefficients
    """
    avg_coeffs = [0,0,0]
    i = 0
    n = len(lines)
    # Weighted average of line coefficients to get smoothed lane lines
    for line in lines:
        i += 1
        avg_coeffs = avg_coeffs + (i * line.coeffs / ((n**2 + n) / 2))

    return avg_coeffs

def convert_coeffs(coeffs, conv_x, conv_y):
    """
    Converts coefficients of 2nd order polynomial. Returns new coefficients
    """
    A = conv_x/(conv_y**2) * coeffs[0]
    B = conv_x/conv_y * coeffs[1]
    C = coeffs[2]

    return np.array([A, B, C])

def cal_cam(path, nx, ny):
    """
    Returns camera calibration parameters. Saves parameters to file
    """
    obj_pts = []    # Coordinate points in real world (3D)
    img_pts = []    # Coordinate points in image space (2D)

    objp = [([x,y,0]) for y in range(ny) for x in range(nx)]
    objp = np.asarray(objp, np.float32)
    cal_images = fnmatch.filter(os.listdir(path), '*.jpg')

    for cal_image in cal_images:
        cal_img = cv2.imread(os.path.join(path, cal_image))
        img_gray = bgr2gray(cal_img)
        ret, corners = cv2.findChessboardCorners(img_gray, (nx,ny), None)

        if ret:
            obj_pts.append(objp)
            img_pts.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, cal_img.shape[1::-1], None, None)
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, cal_img.shape[1::-1], 0)
    # Using newCameraMtx (with alpha = 0) in place of mtx when undistorting images
    # removes empty regions of undistorted images

    np.savez('camera_cal/cal.npz', mtx=mtx, dist=dist, newCameraMtx=newCameraMtx)

    return mtx, dist, newCameraMtx
