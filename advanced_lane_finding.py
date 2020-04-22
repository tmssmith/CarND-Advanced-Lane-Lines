import os
import argparse
import lane_find_utils as utils
import cv2
import numpy as np

class LaneFinder():
    def __init__(self, mtx, dist, newCameraMtx):
        self.mtx = mtx
        self.dist = dist
        self.newCameraMtx = newCameraMtx

    def findlanes(self, img):
        self.img_in = img
        self.img_undist = cv2.undistort(
            self.img_in, self.mtx, self.dist, None, self.newCameraMtx
            )
        self.img_threshold = self.threshold(self.img_undist)
        return self.img_threshold
        self.img_warp = self.warpImage(self.img_threshold)
        lane_px = self.getLanePixels(self.img_warp)
        # left_line = Line(lane_px[0])
        # right_line = Line(lane_px[1])

    def threshold(self, img):
        grad_thresh = self.gradientThreshold(img)
        color_thresh = self.colorThreshold(img)
        img_threshold = grad_thresh | color_thresh

        return img_threshold

    @staticmethod
    def gradientThreshold(img):
        img_gray = utils.bgr2gray(img)
        sobelx_binary = utils.sobel_abs_thresh(img_gray, 'x', 7, (50,180))
        sobely_binary = utils.sobel_abs_thresh(img_gray, 'y', 7, (800,255))
        sobel_mag_binary = utils.sobel_mag_thresh(img_gray, 7, (100,255))
        sobel_dir_binary = utils.sobel_dir_thresh(img_gray, 23, (0.7,1.3))
        grad_binary = \
            (sobelx_binary & sobely_binary) | (sobel_mag_binary & sobel_dir_binary)

        return grad_binary

    @staticmethod
    def colorThreshold(img):
        img_rgb = utils.bgr2rgb(img)
        img_hls = utils.bgr2hls(img)
        r = img_rgb[:,:,0]
        h = img_hls[:,:,0]
        s = img_hls[:,:,2]

        r_binary = cv2.inRange(r, 210, 255)
        h_binary = cv2.inRange(h, 0, 35)
        s_binary = cv2.inRange(s, 135, 255)

        color_binary = (r_binary & h_binary) | s_binary

        return color_binary

    def imgWarp(self, img):
        return img

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Find lane lines in \
        image or video')
    parser.add_argument('input_file', help='File for lane finding')
    parser.add_argument('cal_file', help='Optional: File containing camera \
        calibration matrices', nargs='?', default=None)
    args = parser.parse_args()
    input_fname = os.path.basename(args.input_file)
    input_ext = os.path.splitext(args.input_file)[1]

    # Get camera calibration data
    if args.cal_file == None:
        print('Running camera calibartion...')
        mtx, dist, newCameraMtx = utils.cal_cam('camera_cal', nx=9, ny=6)
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
        frm_rt = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
        frm_dur = int(1000 / frm_rt)
        cv2.namedWindow(input_fname)
        cv2.moveWindow(input_fname, 100,100)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_final = lane.findlanes(frame)
                cv2.imshow(input_fname, frame_final)
                if cv2.waitKey(frm_dur) & 0xFF == ord('q'):
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
    else:
        print('Filetype must be *.jpg or *.mp4')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
