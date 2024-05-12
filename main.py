import mediapipe as mp
import urllib
import sys
from PIL import Image
import cv2
import numpy as np

# import pose estimation model
mp_pose = mp.solutions.pose
# for drawing skeleton
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def get_angle_between_points(p1, p_mid, p2):
    _p1 = np.array(p1)
    _p_mid = np.array(p_mid)
    _p2 = np.array(p2)

    rad = np.arctan2(p2[1]-p_mid[1], p2[0]-p_mid[0]) - np.arctan2(p1[1]-p_mid[1], p1[0]-p_mid[0])
    ang = np.abs(rad*180.0/np.pi)

    if ang > 180.0:
        ang = 360 - ang

    return ang


if __name__ == '__main__':
    file = 'man_squatting_720p.mp4'
    # file = 'man_running.mp4'
    show_only_skeleton = True
    cap = cv2.VideoCapture(file)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\033[33mVideo metadata\033[0m")
    print(f"\033[33mImage resolution:\033[0m {frame_width}x{frame_height}")
    print(f"\033[33mFPS:\033[0m {fps}")
    print(f"\033[33mNumber of frames:\033[0m {frames}")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            elbow = 0
            angle = 0.0

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                print(f"left shoulder: {shoulder}")
                print(f"left elbow: {elbow}")
                print(f"left wrist: {wrist}")

                angle = get_angle_between_points(shoulder, elbow, wrist)
            except:
                pass

            if show_only_skeleton:
                image[:, :] = (0, 0, 0)

            cv2.putText(image, str(int(angle)), np.add(np.multiply(elbow, [frame_width, frame_height]).astype(int),[10,0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

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
