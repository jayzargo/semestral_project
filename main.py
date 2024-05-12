import mediapipe as mp
import urllib
import sys
from PIL import Image
import cv2
import numpy as np


#import pose estimation model
mp_pose = mp.solutions.pose
#for drawing skeleton
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

if __name__ == '__main__':
    #file = 'man_squatting_720p.mp4'
    file = 'man_running.mp4'
    show_only_skeleton = True
    cap = cv2.VideoCapture(file)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                print(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])
            except:
                pass

            if show_only_skeleton:
                image[: , :] = (0, 0, 0)

            mp_drawing.draw_landmarks(image,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()