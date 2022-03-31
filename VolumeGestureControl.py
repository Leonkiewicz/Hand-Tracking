#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from imutils import resize

# pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # volume range -65.25 (min) to 0 (max)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))

cam_width, cam_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

previous_time = 0
current_time = 0
vol_bar = 0
vol_percent = 0
volume_sign = cv2.imread("volumeBar.jpg")
volume_sign = resize(volume_sign, width=30, height=30)
volume_sign_height, volume_sign_width, _ = volume_sign.shape
 
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame")
        continue

    detector = htp.HandDetector(detectConfidence=0.7, trackConfidence=0.9)
    img = detector.find_hands(img)
    hand_landmarks_lst = detector.find_posi(img, draw=False)
    if hand_landmarks_lst:
        print(hand_landmarks_lst[4], hand_landmarks_lst[8])

        cv2.circle(img, (hand_landmarks_lst[4][1], hand_landmarks_lst[4][2]), 10, (0, 0, 0), -1)
        cv2.circle(img, (hand_landmarks_lst[8][1], hand_landmarks_lst[8][2]), 10, (0, 0, 0), -1)
        cv2.line(img, (hand_landmarks_lst[4][1], hand_landmarks_lst[4][2]),
                 (hand_landmarks_lst[8][1], hand_landmarks_lst[8][2]), (0,0,0), 3)

        centre_line_x = (hand_landmarks_lst[4][1] + hand_landmarks_lst[8][1]) // 2
        centre_line_y = (hand_landmarks_lst[4][2] + hand_landmarks_lst[8][2]) // 2
        cv2.circle(img, (centre_line_x, centre_line_y), 10, (0,0,0), -1)

        line_len = math.hypot(hand_landmarks_lst[8][1]-hand_landmarks_lst[4][1],
                              hand_landmarks_lst[8][2]-hand_landmarks_lst[4][2])
        # Hand range - 15-150
        # Volume range - -65.25-0

        vol = np.interp(line_len, [15, 150], [-65.25, 0])
        vol_bar = np.interp(line_len, [15, 150], [10, 400])
        vol_percent = np.interp(line_len, [15, 150], [0, 100])
        print(vol)
        volume.SetMasterVolumeLevel(round(vol), None)

        if line_len < 20:
            cv2.circle(img, (centre_line_x, centre_line_y), 10, (255, 0, 0), -1)

    cv2.rectangle(img, (20, 475), (400, 450), (215, 120, 0), 3)
    cv2.rectangle(img, (20, 475), (int(vol_bar), 450), (215, 120, 0), -1)
    cv2.putText(img, '{}'.format(int(vol_percent)), (420, 470), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

    current_time = time.time()
    fps = 1/(current_time - previous_time)  # calculate frames per second
    previous_time = current_time
    cv2.putText(img, "{}".format(round(fps)), (0, 22), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)  # put fps on screen
    cv2.imshow("Test hand", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




