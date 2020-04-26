import os
import fnmatch
import numpy as np
import cv2

class Window:
    def __init__(self, img, centre_x, base_y, width, height, minpix):
        self.img = img
        self.centre_x = centre_x
        self.p1 = (max(0,centre_x - width//2), max(0,base_y - height))
        self.p2 = (centre_x + width//2, base_y - height)
        self.p3 = (min(img.shape[1], centre_x + width//2), base_y)
        self.p4 = (centre_x - width//2, base_y)
        self.pts = np.array([self.p1,self.p2,self.p3,self.p4], dtype=np.int32)
        self.window = np.array(img[self.p1[1]:self.p3[1], self.p1[0]:self.p3[0]])
        self.minpix = minpix

    def draw_window(self, img, color = (200,200,0)):
        cv2.rectangle(img, self.p1, self.p3, color, 2)

    def get_pixels(self):
        window = self.window
        # window_inds gives positions of nonzero pixels in window frame of reference
        # img_inds gives positions of nonzero pixels in image frame of reference
        window_inds = np.transpose(window.nonzero())
        # Need to swap window_inds columns around as openCV has coordinates as (x,y) but pixels as (row, column)...
        window_inds[:,[1,0]] = window_inds[:,[0,1]]
        self.img_inds = np.add(window_inds, self.p1)

        return self.img_inds

    def draw_pixels(self, img, color = (0,0,255)):
        img_inds = self.img_inds
        img[img_inds[:,1], img_inds[:,0]] = color

    def get_next_centrex(self):
        if len(self.img_inds) >= self.minpix:
            new_centre_x = np.int(np.mean(self.img_inds[:,0]))
            return new_centre_x
        else:
            return self.centre_x


class Line:
    def __init__(self, px, img_height, linetracker):
        self.px = px
        if self.px is not None:
            self.px_cnt = self.px.shape[0]
        else:
            self.px_cnt = 0
        self.coeffs = self.fit_line(linetracker)
        self.curv = self.find_curvature(img_height, 3.7/700, 30/720)
        self.lanepos = self.find_lanepos(img_height)

    def fit_line(self, linetracker):
        if self.px_cnt >= 500:
            self.coeffs = np.polyfit(self.px[:,1], self.px[:,0], 2)
        else:
            self.coeffs = None if not linetracker else linetracker[-1].coeffs

        return self.coeffs

    def find_curvature(self, y, mx, my):
        if self.coeffs is not None:
            A = mx/(my**2) * self.coeffs[0]
            B = mx/my * self.coeffs[1]
            C = self.coeffs[2]
            curv = (1+(2*A*y + B)**2)**(3/2) / abs((2*A))
        else:
            curv = None
        return curv

    def find_lanepos(self, y):
        if self.coeffs is not None:
            lanepos = np.polyval(self.coeffs, y)
        else:
            lanepos = None

        return lanepos


def bgr2rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb

def bgr2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def rgb2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray

def gray2bgr(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return bgr

def bgr2hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    return hls

def get_sobel(img, orient='x', k_size=3):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k_size)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = k_size)
    else:
        print('Orientation must be "x" or "y"')

    return sobel

def sobel_abs_thresh(img, orient='x', k_size=3, threshold=(0,255)):
    sobel = get_sobel(img, orient, k_size)
    abs_sobel = np.absolute(sobel)
    thresh_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mask = cv2.inRange(thresh_sobel, threshold[0], threshold[1])

    return mask

def sobel_mag_thresh(img, k_size=3, threshold=(0,255)):
    sobelx = get_sobel(img, 'x', k_size)
    sobely = get_sobel(img, 'y', k_size)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag_abs = np.uint8(255*sobel_mag/np.max(sobel_mag))
    mask = cv2.inRange(sobel_mag_abs, threshold[0], threshold[1])

    return mask

def sobel_dir_thresh(img, k_size=3, threshold=(0,np.pi/2)):
    sobelx = get_sobel(img, 'x', k_size)
    sobely = get_sobel(img, 'y', k_size)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    mask = cv2.inRange(sobel_dir, threshold[0], threshold[1])

    return mask

def warpImage(img):
    p1 = (593, 467)
    p2 = (722, 467)
    p3 = (1019, 669)
    p4 = (317, 669)
    src = np.array([p1, p2, p3, p4], dtype = np.int32)
    dst = np.array([[p4[0],100],[p3[0],100],p3,p4], dtype=np.int32)

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    img_warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return img_warp, Minv

def gradientThreshold(img):
    img_gray = bgr2gray(img)
    sobelx_binary = sobel_abs_thresh(img_gray, 'x', 7, (50,180))
    sobely_binary = sobel_abs_thresh(img_gray, 'y', 7, (800,255))
    sobel_mag_binary = sobel_mag_thresh(img_gray, 7, (100,255))
    sobel_dir_binary = sobel_dir_thresh(img_gray, 23, (0.7,1.3))
    grad_binary = \
        (sobelx_binary & sobely_binary) | (sobel_mag_binary & sobel_dir_binary)

    return grad_binary

def colorThreshold(img):
    img_rgb = bgr2rgb(img)
    img_hls = bgr2hls(img)
    r = img_rgb[:,:,0]
    h = img_hls[:,:,0]
    s = img_hls[:,:,2]

    r_binary = cv2.inRange(r, 210, 255)
    h_binary = cv2.inRange(h, 0, 35)
    s_binary = cv2.inRange(s, 135, 255)

    color_binary = (r_binary & h_binary) | s_binary

    return color_binary

def sliding_windows(img, peak_x):
    w_width = 200
    w_height = 100
    img_sliding_window = gray2bgr(img.copy())
    img_output = np.zeros_like(img_sliding_window)
    img_height = img.shape[0]
    img_width = img.shape[1]
    lane_pts = []
    lane_color = (0,0,255)
    minpix = 50

    centre_x = peak_x
    base_y = img_height
    for win in range(img_height // w_height + (img_height % w_height > 0)):
        window = Window(img, centre_x, base_y, w_width, w_height, minpix)
        window.draw_window(img_output)
        px_inds = window.get_pixels()
        lane_pts.append(px_inds)
        window.draw_pixels(img_output, lane_color)
        centre_x = window.get_next_centrex()
        base_y -= w_height

    lane_pts = np.concatenate(lane_pts)

    mask = np.zeros_like(img_sliding_window)
    mask[lane_pts[:,1], lane_pts[:,0]] = (255,0,0)
    mask[lane_pts[:,1], lane_pts[:,0]] = (0,255,0)

    return lane_pts, img_output
    # return lane_pts

def search_from_prior(img, line):
    # THIS FUNCTION NEEDS WORK!!!
    img_sfp = gray2bgr(img.copy())
    img_output = np.zeros_like(img_sfp)
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    margin = 100
    lane_pts = []
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    coeffs = line.coeffs
    inds = ((nonzerox >= np.polyval(coeffs, nonzeroy) - margin) &
        (nonzerox <= np.polyval(coeffs, nonzeroy) + margin)).nonzero()[0]
    x = nonzerox[inds]
    y = nonzeroy[inds]
    lane_pts = np.transpose(np.vstack((x,y)))
    img_output[lane_pts[:,1], lane_pts[:,0]] = (0,0,255)

    left_margin = np.polyval(coeffs, plot_y) - margin
    right_margin = np.polyval(coeffs, plot_y) + margin
    left_pts = np.transpose(np.vstack((left_margin, plot_y)))
    right_pts = np.transpose(np.vstack((right_margin, plot_y)))
    cv2.polylines(img_output, np.int32([left_pts]), False, (255,0,0), 4)
    cv2.polylines(img_output, np.int32([right_pts]), False, (255,0,0), 4)

    return lane_pts, img_output

def draw_lane(img, leftlines, rightlines, Minv):
    mx = 3.7/700
    my = 30/720
    left_coeffs = wgtd_avg(leftlines)
    right_coeffs = wgtd_avg(rightlines)

    mid_coeffs = (left_coeffs + right_coeffs) / 2
    # NEEDS SORTING PROPERLY

    A = mx/(my**2) * mid_coeffs[0]
    B = mx/my * mid_coeffs[1]
    C = mid_coeffs[2]
    curv = (1+(2*A*img.shape[0] + B)**2)**(3/2) / abs((2*A))

    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])

    leftline = leftlines[-1]
    rightline = rightlines[-1]
    left_x = np.polyval(left_coeffs, plot_y)
    right_x = np.polyval(right_coeffs, plot_y)
    left_pts = np.transpose(np.vstack((left_x, plot_y)))
    right_pts = np.transpose(np.vstack((right_x, plot_y)))

    # Create an image to draw the lane guide on
    canvas = np.zeros_like(img)
    pts = np.vstack((left_pts, right_pts[::-1]))
    cv2.fillPoly(canvas, np.int32([pts]), (0,255, 0))

    # Use inverse perspective matrix (Minv) to warp canvas back to original image space
    lane_warp = cv2.warpPerspective(canvas, Minv, (img.shape[1], img.shape[0]))
    img_out = cv2.addWeighted(img, 1, lane_warp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    lanecentre = int((leftline.lanepos + rightline.lanepos) / 2)
    veh_center = int(img_out.shape[1]/2)
    cv2.line(img_out, (lanecentre, img_out.shape[0]),(lanecentre, img_out.shape[0] - 50), (0,0,0),3)
    cv2.line(img_out, (veh_center, img_out.shape[0]),(veh_center, img_out.shape[0]- 40),(255,0,0),3)

    # lanecurv = (leftline.curv + rightline.curv)/2
    laneoffset = (lanecentre - img_out.shape[1]/2) * (3.7 / 700)
    cv2.putText(img_out, 'Lane curvature: {:.3f} km'.format(curv / 1000), (50,50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_out, 'Lane position: {:.2f} m'.format(laneoffset), (50,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_out

def wgtd_avg(lines):
    avg_coeffs = [0,0,0]
    i = 0
    n = len(lines)
    # Weighted average of line coefficients to get smoothed lane lines
    for line in lines:
        i += 1
        avg_coeffs = avg_coeffs + (i * line.coeffs / ((n**2 + n) / 2))

    return avg_coeffs

def cal_cam(path, nx, ny):
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
