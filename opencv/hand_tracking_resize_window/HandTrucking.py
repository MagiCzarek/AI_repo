import cv2
import mediapipe as mp
import time


# noinspection PyUnresolvedReferences

def calc_fps(prev_time):
    curr_time = time.time()
    frame_per_second = 1 / (curr_time - prev_time)
    prev_time = curr_time
    return prev_time, frame_per_second


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLandMarks in result.multi_hand_landmarks:
                for id, lm in enumerate(handLandMarks.landmark):
                    # print(id, lm)
                    height, width, channels = frame.shape
                    pos_x, pos_y = int(lm.x * width), int(lm.y * height)
                    print(id, pos_x,pos_y)
                    if id == 0:
                        cv2.circle(frame, (pos_x,pos_y), 20 ,(255,255,0),cv2.FILLED)
                mp_draw.draw_landmarks(frame, handLandMarks, mp_hands.HAND_CONNECTIONS)

        prev_time, frame_per_second = calc_fps(prev_time)

        cv2.putText(frame, str(int(frame_per_second)), (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 3,
                    (255, 255, 0), 3)

        cv2.imshow("img", frame)
        key = cv2.waitKey(2) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
