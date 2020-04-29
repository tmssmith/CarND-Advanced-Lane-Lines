import os
import argparse
import lane_find_utils as utils
import cv2
import numpy as np
from parameters import *

class LaneFinder():
    """
    Finds lane lines in image and tracks lan lines from previous images

    Args:
        mtx: Camera calibration matrix
        dist: Camer distortion coefficients
        newCameraMtx: New camera matrix

    Attributes:
        leftlinetracker (list): left lane lines from previous images
        rightlinetracker (list) right lane lines from previous images

    """

    def __init__(self, mtx, dist, newCameraMtx):
        self.mtx = mtx
        self.dist = dist
        self.newCameraMtx = newCameraMtx
        self.leftlinetracker = []
        self.rightlinetracker = []

    def findlanes(self, img):
        """
        Co-ordinates lane finding pipeline. Returns image with lane overaly
        """
        mtx, dist, newCameraMtx = self.mtx, self.dist, self.newCameraMtx
        img_in = img
        img_undist = cv2.undistort(img_in, mtx, dist, None, newCameraMtx)
        img_warp, Minv = utils.warpImage(img_undist)
        img_threshold = utils.threshold(img_warp)
        lane_px = self.getLanePixels(img_threshold)
        self.left_line = utils.Line(lane_px[0], self.leftlinetracker)
        self.right_line = utils.Line(lane_px[1], self.rightlinetracker)
        self.leftlinetracker.append(self.left_line)
        self.rightlinetracker.append(self.right_line)
        if len(self.leftlinetracker) > WGTD_MVNG_AVG_LEN:
            self.leftlinetracker = self.leftlinetracker[1:]
        if len(self.rightlinetracker) > WGTD_MVNG_AVG_LEN:
            self.rightlinetracker = self.rightlinetracker[1:]
        img_final = utils.draw_lane(img_undist, self.leftlinetracker, self.rightlinetracker, Minv)

        return img_final

    def getLanePixels(self, img):
        """
        Returns pixels identified as being in left and right lane lines
        """
        img_hist = img[img.shape[0]//2:,:]
        hist_y = np.sum(img_hist, 0)
        hist_x = np.arange(0, img_hist.shape[1], 1)
        mid_x = len(hist_y) // 2
        leftpeak_x = np.argmax(hist_y[:mid_x])
        rightpeak_x = np.argmax(hist_y[mid_x:]) + mid_x

        if not self.leftlinetracker:
            left_lane_pts = utils.sliding_windows(img, leftpeak_x)
        elif self.leftlinetracker[-1].px_cnt < MINPIX_PRIOR:
            left_lane_pts = utils.sliding_windows(img, leftpeak_x)
        elif self.leftlinetracker[-1].coeffs[-1] == self.rightlinetracker[-1].coeffs[-1]:
            # Protect against the same line being detected twice
            left_lane_pts = utils.sliding_windows(img, leftpeak_x)
        else:
            left_lane_pts = utils.search_from_prior(img, self.left_line)

        if not self.rightlinetracker:
            right_lane_pts = utils.sliding_windows(img, rightpeak_x)
        elif self.rightlinetracker[-1].px_cnt < MINPIX_PRIOR:
            right_lane_pts = utils.sliding_windows(img, rightpeak_x)
        elif self.leftlinetracker[-1].coeffs[-1] == self.rightlinetracker[-1].coeffs[-1]:
            # Protect against the same line being detected twice
            right_lane_pts = utils.sliding_windows(img, rightpeak_x)
        else:
            right_lane_pts = utils.search_from_prior(img, self.right_line)

        return (left_lane_pts, right_lane_pts)

def main():
    """Sets up LaneFinder and runs findlanes for image or video frames"""
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Find lane lines in \
        image or video')
    parser.add_argument('input_file', help='File for lane finding')
    parser.add_argument('cal_file', help='Optional: File containing camera \
        calibration matrices', nargs='?', default=None)
    args = parser.parse_args()
    input_fname = os.path.basename(args.input_file)
    input_ext = os.path.splitext(args.input_file)[1]
    input_name = os.path.splitext(input_fname)[0]

    # Get camera calibration data
    if args.cal_file == None:
        print('Running camera calibartion...')
        mtx, dist, newCameraMtx = utils.cal_cam('camera_cal', NX, NY)
    else:
        print('Loading calibration file...')
        cal_file = np.load(args.cal_file)
        if cal_file is not None:
            mtx = cal_file['mtx']
            dist = cal_file['dist']
            newCameraMtx = cal_file['newCameraMtx']
        else:
            print('Error opening camera calibration file')

    lane = LaneFinder(mtx, dist, newCameraMtx)

    # Open image or video file, perform advanced lane finding
    if input_ext == '.jpg':
        print('Processing input image...')
        img = cv2.imread(args.input_file)
        if img is not None:
            cv2.namedWindow(input_fname)
            cv2.moveWindow(input_fname, 100,100)
            while True:
                img_final = lane.findlanes(img)
                cv2.imshow(input_fname, img_final)
                if cv2.waitKey(40) & 0xFF == ord('q'):
                    break
        else:
            print('Error opening image')
    elif input_ext == '.mp4':
        print('Processing input video...')
        cap = cv2.VideoCapture(args.input_file)
        if cap.isOpened() == False:
            print("Error opening video")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
        frm_dur = int(1000 / fps)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output_videos/' + input_name + '_output.mp4', \
            cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))
        cv2.namedWindow(input_fname)
        cv2.moveWindow(input_fname, 100,100)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_final = lane.findlanes(frame)
                cv2.imshow(input_fname, frame_final)
                out.write(frame_final)
                if cv2.waitKey(frm_dur) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
    else:
        print('Filetype must be *.jpg or *.mp4')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
else:
    print('This module is intended to be the main module and should not be \
        imported by another module')
