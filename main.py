import mediapipe as mp
import urllib
import sys
from PIL import Image
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


if __name__ == '__main__':
    #file = 'man_walking.jpg'
    file = 'man_sitting.jpg'
    #file = 'man_crawling.jpg'

    # Create a MediaPipe `Pose` object
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=2,
                      enable_segmentation=True) as pose:
        # Read the file in and get dims
        image = cv2.imread(file)

        # Convert the BGR image to RGB and then process with the `Pose` object.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Copy the iamge
    annotated_image = image.copy()

    # Draw pose, left and right hands, and face landmarks on the image with drawing specification defaults.
    mp_drawing.draw_landmarks(annotated_image,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Save image with drawing
    #filename = "pose_wireframe.png"
    #cv2.imwrite(filename, annotated_image)

    cv2.imshow('image',annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()