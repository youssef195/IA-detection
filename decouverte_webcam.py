import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose




cap= cv2.VideoCapture(0)

while cap.isOpened():
    
    ret , frame=cap.read()
    cv2.imshow('mdiapipe',frame)

    if cv2.waitKey(10) &0xFF== ord('q'):
        break
cap.release()
cap.destroyALLWindows()