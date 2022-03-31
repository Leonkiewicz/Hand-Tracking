#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

previous_time = 0
current_time = 0



with mp_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB color space, "hands" object only uses RGB images
        img.flags.writeable = False  # to improve performance, mark the img as not writeable to pass by reference
        results = hands.process(img_RGB)  # Processes an RGB image and returns the hand landmarks and handedness (left or right hand) of each detected hand.
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # print(idx, landmark)
                    height, width, _ = img.shape
                    x_posi, y_posi = int(landmark.x*width), int(landmark.y*height)  # position of  x and y in pixels
                    print(idx, x_posi, y_posi)

                    if idx == 0:
                        cv2.circle(img, (x_posi, y_posi), 10, (0, 0, 0), -1)

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hand.HAND_CONNECTIONS)

        current_time = time.time()
        fps = 1/(current_time-previous_time)  # calculate frames per second
        previous_time = current_time

        cv2.putText(img, "{}".format(round(fps)), (0, 22), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2) # put fps on screen
        cv2.imshow("MediaPipe Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

