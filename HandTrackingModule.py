#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, max_num_of_hands=2, detectConfidence=0.7, trackConfidence=0.7):
        self.mode = mode
        self.max_num_of_hands = max_num_of_hands
        self.detectConfidence = detectConfidence
        self.trackConfidence = trackConfidence

        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(self.mode, max_num_of_hands, detectConfidence, trackConfidence)
        self.mp_drawing = mp.solutions.drawing_utils


    def find_hands(self, img, draw=True):
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB color space, "hands" object only uses RGB images
            img.flags.writeable = False  # to improve performance, mark the img as not writeable to pass by reference
            self.results = self.hands.process(img_RGB)  # Processes an RGB image and returns the hand landmarks and handedness (left or right hand) of each detected hand.
            # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks is not None:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    if draw:
                        self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hand.HAND_CONNECTIONS)
            return img


    def find_posi(self, img, handNumber=0, draw=True):      # maybe create a helper function here
        hand_landmarks_lst = []
        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = img.shape
                    x_posi, y_posi = int(landmark.x * width), int(landmark.y * height)  # position of  x and y in pixels

                    hand_landmarks_lst.append([idx, x_posi, y_posi])
                    if draw:
                        cv2.circle(img, (x_posi, y_posi), 10, (0, 0, 0), -1)

            return hand_landmarks_lst
def main():
    previous_time = 0
    current_time = 0
    detector = HandDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        img = detector.find_hands(img)
        hand_landmarks_lst = detector.find_posi(img)
        if hand_landmarks_lst:
            print(hand_landmarks_lst[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)  # calculate frames per second
        previous_time = current_time

        cv2.putText(img, "{}".format(round(fps)), (0, 22), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)  # put fps on screen
        cv2.imshow("MediaPipe Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
