import mediapipe as mp
import urllib
import sys
import cv2
import numpy as np

# import pose estimation model
mp_pose = mp.solutions.pose
# for drawing skeleton
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def populate_body_angles(landmarks, features):
    features['positions']['left_shoulder'] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    features['positions']['left_elbow'] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    features['positions']['left_wrist'] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    features['positions']['left_hip'] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    features['positions']['left_knee'] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    features['positions']['left_ankle'] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    features['positions']['right_shoulder'] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    features['positions']['right_elbow'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    features['positions']['right_wrist'] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    features['positions']['right_hip'] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    features['positions']['right_knee'] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    features['positions']['right_ankle'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    features['angles']['left_shoulder'] = get_angle_between_points(
        features['positions']['left_hip'],
        features['positions']['left_shoulder'],
        features['positions']['left_elbow']
    )
    features['angles']['left_elbow'] = get_angle_between_points(
        features['positions']['left_shoulder'],
        features['positions']['left_elbow'],
        features['positions']['left_wrist']
    )
    features['angles']['left_hip'] = get_angle_between_points(
        features['positions']['right_hip'],
        features['positions']['left_hip'],
        features['positions']['left_knee']
    )
    features['angles']['left_knee'] = get_angle_between_points(
        features['positions']['left_hip'],
        features['positions']['left_knee'],
        features['positions']['left_ankle']
    )
    features['angles']['right_shoulder'] = get_angle_between_points(
        features['positions']['right_hip'],
        features['positions']['right_shoulder'],
        features['positions']['right_elbow']
    )
    features['angles']['right_elbow'] = get_angle_between_points(
        features['positions']['right_shoulder'],
        features['positions']['right_elbow'],
        features['positions']['right_wrist']
    )
    features['angles']['right_hip'] = get_angle_between_points(
        features['positions']['left_hip'],
        features['positions']['right_hip'],
        features['positions']['right_knee']
    )
    features['angles']['right_knee'] = get_angle_between_points(
        features['positions']['right_hip'],
        features['positions']['right_knee'],
        features['positions']['right_ankle']
    )


def draw_features(image, features, frame_width, frame_height):
    cv2.putText(image, str(int(features['angles']['left_shoulder'])),
                np.add(np.multiply(features['positions']['left_shoulder'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['left_elbow'])),
                np.add(np.multiply(features['positions']['left_elbow'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['left_hip'])),
                np.add(np.multiply(features['positions']['left_hip'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['left_knee'])),
                np.add(np.multiply(features['positions']['left_knee'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['right_shoulder'])),
                np.add(np.multiply(features['positions']['right_shoulder'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['right_elbow'])),
                np.add(np.multiply(features['positions']['right_elbow'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['right_hip'])),
                np.add(np.multiply(features['positions']['right_hip'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, str(int(features['angles']['right_knee'])),
                np.add(np.multiply(features['positions']['right_knee'], [frame_width, frame_height]).astype(int),
                       [10, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def get_angle_between_points(p1, p_mid, p2):
    _p1 = np.array(p1)
    _p_mid = np.array(p_mid)
    _p2 = np.array(p2)

    rad = np.arctan2(p2[1] - p_mid[1], p2[0] - p_mid[0]) - np.arctan2(p1[1] - p_mid[1], p1[0] - p_mid[0])
    ang = np.abs(rad * 180.0 / np.pi)

    if ang > 180.0:
        ang = 360 - ang

    return ang


def extract_features(videofile):
    body_features = {
        'angles': {
            'left_shoulder': 0.0,
            'left_elbow': 0.0,
            'right_shoulder': 0.0,
            'right_elbow': 0.0,
            'left_hip': 0.0,
            'left_knee': 0.0,
            'right_hip': 0.0,
            'right_knee': 0.0
        },
        'positions': {
            'left_shoulder': 0.0,
            'left_elbow': 0.0,
            'right_shoulder': 0.0,
            'right_elbow': 0.0,
            'left_hip': 0.0,
            'left_knee': 0.0,
            'right_hip': 0.0,
            'right_knee': 0.0,
            'left_ankle': 0.0,
            'left_wrist': 0.0,
            'right_ankle': 0.0,
            'right_wrist': 0.0
        }
    }

    show_background = False
    show_features = True
    show_skeleton = True
    cap = cv2.VideoCapture(videofile)

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
                populate_body_angles(landmarks, body_features)
            except:
                pass

            if not show_background:
                image[:, :] = (0, 0, 0)

            if show_features:
                draw_features(image, body_features, frame_width, frame_height)

            if show_skeleton:
                mp_drawing.draw_landmarks(image,
                                          results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS,
                                          None,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    file = 'man_squatting_720p.mp4'
    # file = 'man_running.mp4'
    extract_features(file)
