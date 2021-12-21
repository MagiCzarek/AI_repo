import cv2
import matplotlib.pyplot as plt
import inspect
import numpy as np
import argparse
import sys

"""
Cezary_Boguszewski object_tracking

Script for tracking object from images on input

arg parse commented because of testing in evironment

"""


# def pars_arguments():
#     parser = argparse.ArgumentParser(description='This is a script for detect and track the object on video ')
#     parser.add_argument('-v', '--video', type=str, required=True,
#                         nargs='+', help='video path')
#     parser.add_argument('-i', '--image', type=str, required=True,
#                         nargs='+', help='You can add images path')
#
#     return parser.parse_args()


# function for rescaling image
def rescale_image(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# function for showing image
def show_img(img, bw=False):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='Greys_r' if bw else None)
    plt.show()


# function to refactor color space
def change_colorspace_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def print_interesting_members(obj):
    for name, value in inspect.getmembers(obj):
        try:
            float(value)
            print(f'{name} -> {value}')
        except Exception:
            pass


# function for executing ORB ALGORITHM
def execute_ORB(img_1, img_2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img_1, None)
    kp2, des2 = orb.detectAndCompute(img_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return img_1, img_2, kp1, kp2, des1, des2, matches


# function for 2 images and visualize this
def match_images(img_1, img_2):
    min_distance_to_pair = 35
    min_matches_to_pair = 4
    good_matches = []
    img_1, img_2, kp1, kp2, des1, des2, matches = execute_ORB(img_1, img_2)

    for m in matches:
        if m.distance <= min_distance_to_pair:
            good_matches.append(m)

    if len(good_matches) >= min_matches_to_pair:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        print("Matches found = {}".format(len(good_matches)))

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        match_mask = mask.ravel().tolist()
        h, w = img_1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img_2 = cv2.polylines(img_2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        match_mask = None
        print("The image not matches")

    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=None,
                       matchesMask=match_mask,
                       flags=2)

    matches_vis = cv2.drawMatches(img_1, kp1, img_2, kp2, good_matches, None, **draw_params)
    show_img(matches_vis)

    return good_matches


# function for video service
def video_run(images, video):
    number_of_images = len(images)
    print(number_of_images)
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    while ret:
        if frame is not None:
            frame = change_colorspace_to_gray(frame)
            rescale_image(frame, 30)
            for i in range(number_of_images):
                match = []
                matches = match_images(images[i], frame)
                for j in range(len(matches)):
                    match.append(matches[j])
                    print(f'Distance from matches = {matches[j].distance}')
            print_interesting_members(match)
        ret, frame = cap.read()


# testmain
def main():
    images = []
    video_path = '../../vod.mp4'

    for i in range(1, 3):
        img_path = 'E:\pythonProject3\img{}.jpg'.format(i)
        tmp_img = cv2.imread(img_path)
        tmp_img = change_colorspace_to_gray(tmp_img)
        tmp_img = rescale_image(tmp_img, 30)
        images.append(tmp_img)
    video_run(images, video_path)


# main with args from argparse
def main2(input_video, input_images):
    images = []
    images = input_images

    for i in range(len(images)):
        tmp_img = images[i]
        tmp_img = cv2.imread(tmp_img)
        tmp_img = change_colorspace_to_gray(tmp_img)
        tmp_img = rescale_image(tmp_img, 30)
        images[i] = tmp_img

    video_run(images, input_video)


if __name__ == '__main__':
    main()

    # try:
    #     args = pars_arguments()
    #
    #     main2(str(args.video), args.image)
    # except SystemExit:
    #     exc = sys.exc_info()[1]
    #     print(exc)
