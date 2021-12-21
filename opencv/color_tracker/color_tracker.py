import argparse
import sys
import numpy as np
import cv2
"""


This is my first script from open-cv, so it isnt't clean at all.

"""

def pars_arguments():
    parser = argparse.ArgumentParser(description='Script for tracking colors on video')
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Input video')
    parser.add_argument('--foo', nargs='?', help='This is a script for tracking colors on video or from camera. If '
                                                 ' you want to upload video you need to '
                                                 'execute script by writing'
                                                 ' python color_tracker.py -i "path"'
                                                 'You can test it on example by writing:'
                                                 'python color_tracker.py -i rgb.mp4    '
                                                 'If you want to execute script by loading video from camera execute this'
                                                 'without arguments.'
                                                 ' IN PROGRAMME :'
                                                 'space = pause,'
                                                 'unpausing by any button, '
                                                 'q = quit,'
                                                 'right click = change color '
                                                 'c = reqtangle  and the circle on middle of item,'
                                                 'tracking but you must unpause 1st, the circle is added because'
                                                 'rectangle not always working'
                                                 'w =  on this button u can see only this color mask' )
    # parser.add_argument('bar', nargs='+', help='This is a script for tracking colors on video or from camera. If '
    #                                            'If you want to upload video you need to'
    #                                            'execute script by writing'
    #                                            ' python color_tracker.py -i "path"'
    #                                            'You can test it on example by writing:'
    #                                            'python color_tracker.py -i rgb.mp4    '
    #                                            'If you want to execute script by loading video from camera execute this'
    #                                            'without arguments.')
    return parser.parse_args()


def on_click_hsv(event, x, y, flags, param):
    global lower_limit
    global upper_limit
    global color
    global colorB
    global colorG
    global colorR
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorB = param[y, x, 0]
        colorG = param[y, x, 1]
        colorR = param[y, x, 2]

        print("Value Red:", colorR)
        print("Value Green:", colorG)
        print("Value Blue:", colorB)

        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        h = hsv[y, x, 0]
        s = hsv[y, x, 1]
        v = hsv[y, x, 2]
        # if(lower_limit is None):
        lower_limit = np.array([h - 15, s - 80, v-70])
        # elif(upper_limit is None):
        upper_limit = np.array([h + 15, s + 80, v+70])


def maskImage(lower_limit, upper_limit, frame):
    global ret
    global thresh
    global color_frame
    mask = cv2.inRange(hsv, lower_limit, upper_limit)

    color_frame = cv2.bitwise_and(frame, frame, mask=mask)

    color_frame = cv2.erode(color_frame, kernel, iterations=3)
    color_frame = cv2.dilate(color_frame, kernel, iterations=3)

    frame_gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(frame_gray, 0, 255, 0)
    return color_frame, thresh, ret


def keys_switcher(key):
    global show_captured_color
    global show_frame
    global key_return

    if key == ord('q'):
        return sys.exit()
    elif key == ord('w'):
        show_captured_color = not show_captured_color
        return show_captured_color
    elif key == ord('c'):
        show_frame = not show_frame
        return show_frame
    elif key == ord(' '):
        cv2.waitKey(-1)


def main(input_file):
    print(input_file)
    if input_file == 0:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(input_file)

    global lower_limit
    global upper_limit
    lower_limit = None
    upper_limit = None

    global default_limit
    default_limit = np.array([0, 0, 0])

    global show_captured_color
    show_captured_color = False
    global show_frame
    show_frame = False

    global key

    global colorB
    global colorG
    global colorR

    frame_name = input_file.split('.')[0]

    global kernel
    kernel = np.ones((2, 2), np.uint8)

    video_view, frame = video.read()

    while video_view:
        global hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if (upper_limit is None and lower_limit is None):
            color_frame, thresh, ret = maskImage(default_limit, default_limit, frame)
        else:
            color_frame, thresh, ret = maskImage(lower_limit, upper_limit, frame)

        M = cv2.moments(thresh)

        cv2.setMouseCallback('rgb', on_click_hsv, frame)

        if show_captured_color:
            cv2.imshow(frame_name, color_frame)


        else:
            if show_frame:
                if M["m00"] != 0:
                    contours, hierarchy = cv2.findContours(thresh, 1, 2)
                    cnt = contours[0]
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                else:
                    cX, cY = 0, 0


                cv2.circle(frame, (cX, cY), 1, (255, 255, 255), -1)
                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (int(colorB), int(colorG), int(colorR)), 2)

            cv2.imshow(frame_name, frame)

        key = cv2.waitKey(1)
        keys_switcher(key)

        video_view, frame = video.read()

    video.release()


if __name__ == '__main__':
    try:
        args = pars_arguments()
        print(type(args.input_video))
        if args is None:
            main(0)
        else:
            main(args.input_video)

    except SystemExit:
        exc = sys.exc_info()[1]
        print(exc)
