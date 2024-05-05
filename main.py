import mediapipe as mp
import urllib
import sys
from PIL import Image
import cv2
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


if __name__ == '__main__':
    poselandmarks_list = []

    #file = 'man_squatting_720p.mp4'
    file = 'man_running.mp4'
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        capture = cv2.VideoCapture(file)

        if not capture.isOpened():
            print("Error opening video file")
            raise TypeError

        frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a NumPy array to store the pose data as before
        # The shape is 3x33x144 - 3D XYZ data for 33 landmarks across 144 frames
        data = np.empty((2, 33, frames))

        frame_num = 0

        size = (frame_width, frame_height)

        while capture.isOpened():
            ret, image = capture.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            annotated_image = image.copy()
            results = pose.process(image)

            mp_drawing.draw_landmarks(annotated_image,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            landmarks = results.pose_world_landmarks.landmark
            for i in range(len(mp_pose.PoseLandmark)):
                data[:, i, frame_num] = (landmarks[i].x, landmarks[i].y)

            frame_num += 1

            cv2.imshow('Frame', annotated_image)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Close the video file
        capture.release()
        cv2.destroyAllWindows()
