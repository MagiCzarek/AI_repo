import cv2
import mediapipe as mp
import time


# noinspection PyUnresolvedReferences
class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence,
                                         self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def calc_fps(self,prev_time):
        curr_time = time.time()
        frame_per_second = 1 / (curr_time - prev_time)
        prev_time = curr_time
        return prev_time, frame_per_second




    def detect_hands(self, frame, show=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:

            for hand_land_marks in self.result.multi_hand_landmarks:
                if show:
                    self.mp_draw.draw_landmarks(frame, hand_land_marks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_position(self, frame, hand_number=0, show=True):

        pos_list = []

        if self.result.multi_hand_landmarks:
            current_hand = self.result.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(current_hand.landmark):
                height, width, channels = frame.shape
                pos_x, pos_y = int(lm.x * width), int(lm.y * height)
                pos_list.append([id, pos_x, pos_y])
                if show:
                    cv2.circle(frame, (pos_x, pos_y), 20, (255, 255, 0), cv2.FILLED)
        return pos_list




def main():
    cap = cv2.VideoCapture(0)
    hand_detector = handDetector()
    prev_time = 0

    while True:
        ret, frame = cap.read()
        frame = hand_detector.detect_hands(frame)
        pos_list = hand_detector.get_position(frame)

        if len(pos_list) != 0:
             print(pos_list[2])

        prev_time, frame_per_second = hand_detector.calc_fps(prev_time)

        cv2.putText(frame, str(int(frame_per_second)), (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 3,
                    (255, 255, 0), 3)

        cv2.imshow("img", frame)
        key = cv2.waitKey(2) & 0xff
        if key == 27:
            break


if __name__ == '__main__':
    main()
