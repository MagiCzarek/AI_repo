import argparse
import sys
import numpy as np
import cv2
"""
This is my first script in python for tracking color on image
"""

def pars_arguments():
    parser = argparse.ArgumentParser(description='Script for detecting red balls')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='input')
    parser.add_argument('-f', '--fps', type=int, required=True, help='Frames per second')
    return parser.parse_args()


def main(input_file, fps):
    video = cv2.VideoCapture(input_file)

    if video is None:
        print('Unable to open file {}'.format(input_file))
        sys.exit()

    frame_name = input_file.split('.')[0]
    kernel = np.ones((5, 5), np.uint8)


    rec, frame = video.read()
    while rec:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([2, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([210, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask0 + mask1
        frame_red = cv2.bitwise_and(frame, frame, mask=mask)

        frame_red = cv2.erode(frame_red, kernel, iterations=3)
        frame_red = cv2.dilate(frame_red, kernel, iterations=3)

        frame_gray = cv2.cvtColor(frame_red, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame_gray, 0, 255, 0)

        M = cv2.moments(thresh)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)
        cv2.imshow(frame_name, frame_red)
        cv2.imshow("frame_name", frame)


        key = cv2.waitKey(1)
        if key == ord('q'):
            sys.exit()


        rec, frame = video.read()

    video.release()


if __name__ == '__main__':
    args = pars_arguments()
    main(args.input_file, args.fps)
