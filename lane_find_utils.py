import os
import fnmatch

import numpy as np
import cv2

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
