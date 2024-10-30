import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# 상수 정의
PADDING = 30
FACE_SIZE = (1024, 1024)
FINAL_SIZE = (256, 128)
threshold_X = 0.08
threshold_Y = 0.16
threshold_close = 0.32

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_face_detection = mp.solutions.face_detection

# 입력 폴더와 출력 폴더의 기본 경로
base_input_folder = r'C:\Users\SU\Desktop\DOT DELETE\normal_team'
base_output_folder = r'C:\Users\SU\Desktop\DOT DELETE\testing'

# 눈의 랜드마크 인덱스 정의
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_IRIS_CENTER_INDEX = 468
RIGHT_IRIS_CENTER_INDEX = 473


def crop_face(image, detection):
    """얼굴 이미지에서 얼굴 부분을 크롭"""
    ih, iw, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    x_min = int(max(bboxC.xmin * iw, 0))
    y_min = int(max(bboxC.ymin * ih, 0))
    x_max = int(min((bboxC.xmin + bboxC.width) * iw, iw))
    y_max = int(min((bboxC.ymin + bboxC.height) * ih, ih))

    if x_min >= x_max or y_min >= y_max:
        return None
    cropped_face = image[y_min:y_max, x_min:x_max]
    return cv2.resize(cropped_face, FACE_SIZE) if cropped_face.size > 0 else None


def determine_iris_position(eye_center, iris_center, eye_aspect_ratio):
    """눈동자의 위치에 따라 눈의 방향을 결정"""
    x_ratio = (iris_center[0] - eye_center[0]) / (FINAL_SIZE[0] / 2)
    y_ratio = (iris_center[1] - eye_center[1]) / (FINAL_SIZE[1] / 2)

    if abs(x_ratio) > threshold_X:
        return 'Left' if x_ratio < 0 else 'Right'
    if eye_aspect_ratio < threshold_close:
        return 'Down'
    if abs(y_ratio) > threshold_Y:
        return 'Up' if y_ratio < 0 else 'Down'
    return 'Center'


def detect_and_crop_eyes(face_image):
    """얼굴 이미지에서 눈을 검출하고 크롭"""
    resized_image = cv2.resize(face_image, FACE_SIZE)

    results = face_mesh.process(resized_image)
    if not results.multi_face_landmarks:
        return None, None, None, None

    face_landmarks = results.multi_face_landmarks[0]

    # 눈 중심 좌표 계산
    left_eye_landmarks = [face_landmarks.landmark[idx] for idx in LEFT_EYE_INDICES]
    right_eye_landmarks = [face_landmarks.landmark[idx] for idx in RIGHT_EYE_INDICES]

    left_eye_x = sum([lm.x for lm in left_eye_landmarks]) / len(left_eye_landmarks) * resized_image.shape[1]
    left_eye_y = sum([lm.y for lm in left_eye_landmarks]) / len(left_eye_landmarks) * resized_image.shape[0]
    left_eye_center = int(left_eye_x), int(left_eye_y)

    right_eye_x = sum([lm.x for lm in right_eye_landmarks]) / len(right_eye_landmarks) * resized_image.shape[1]
    right_eye_y = sum([lm.y for lm in right_eye_landmarks]) / len(right_eye_landmarks) * resized_image.shape[0]
    right_eye_center = int(right_eye_x), int(right_eye_y)

    left_iris_center = (int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                        int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))
    right_iris_center = (int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                         int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))

    left_eye_points = np.array([(int(face_landmarks.landmark[idx].x * resized_image.shape[1]),
                                 int(face_landmarks.landmark[idx].y * resized_image.shape[0])) for idx in LEFT_EYE_INDICES])
    right_eye_points = np.array([(int(face_landmarks.landmark[idx].x * resized_image.shape[1]),
                                  int(face_landmarks.landmark[idx].y * resized_image.shape[0])) for idx in RIGHT_EYE_INDICES])

    # 눈의 세로/가로 비율을 직접 계산
    left_eye_rect = cv2.boundingRect(left_eye_points)
    _, _, w_left, h_left = left_eye_rect
    left_eye_aspect_ratio = h_left / w_left if w_left != 0 else 0

    right_eye_rect = cv2.boundingRect(right_eye_points)
    _, _, w_right, h_right = right_eye_rect
    right_eye_aspect_ratio = h_right / w_right if w_right != 0 else 0

    left_iris_status = determine_iris_position(left_eye_center, left_iris_center, left_eye_aspect_ratio)
    right_iris_status = determine_iris_position(right_eye_center, right_iris_center, right_eye_aspect_ratio)

    # 눈 이미지 추출 및 크기 조정
    x_left, y_left, w_left, h_left = cv2.boundingRect(left_eye_points)
    left_eye_image = resized_image[max(y_left - PADDING, 0):min(y_left + h_left + PADDING, resized_image.shape[0]),
                                   max(x_left - PADDING, 0):min(x_left + w_left + PADDING, resized_image.shape[1])]
    left_eye_image = cv2.resize(left_eye_image, FINAL_SIZE)

    x_right, y_right, w_right, h_right = cv2.boundingRect(right_eye_points)
    right_eye_image = resized_image[max(y_right - PADDING, 0):min(y_right + h_right + PADDING, resized_image.shape[0]),
                                    max(x_right - PADDING, 0):min(x_right + w_right + PADDING, resized_image.shape[1])]
    right_eye_image = cv2.resize(right_eye_image, FINAL_SIZE)

    return left_eye_image, right_eye_image, left_iris_status, right_iris_status


def process_image(image_path, output_folder, username):
    """이미지를 처리하고 결과를 저장"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_results = face_detection.process(image_rgb)
        if not detection_results.detections:
            print(f"No face detected in {image_path}")
            return

        cropped_face = crop_face(image, detection_results.detections[0])
        if cropped_face is None:
            print(f"Error cropping face in {image_path}")
            return

        left_eye, right_eye, left_iris_status, right_iris_status = detect_and_crop_eyes(cropped_face)
        if left_eye is None or right_eye is None:
            print(f"Could not detect both eyes in {image_path}")
            return

        # 왼쪽 눈과 오른쪽 눈 폴더 설정
        left_eye_folder = os.path.join(output_folder, "left_eye")
        right_eye_folder = os.path.join(output_folder, "right_eye")
        combined_eye_folder = os.path.join(output_folder, "combined_eye")
        os.makedirs(left_eye_folder, exist_ok=True)
        os.makedirs(right_eye_folder, exist_ok=True)
        os.makedirs(combined_eye_folder, exist_ok=True)

        # 눈 방향에 따른 하위 폴더 생성
        for direction in ['Left', 'Right', 'Up', 'Down', 'Center']:
            os.makedirs(os.path.join(left_eye_folder, direction), exist_ok=True)
            os.makedirs(os.path.join(right_eye_folder, direction), exist_ok=True)

        # 사용자명 앞에 추가하여 저장
        left_eye_filename = os.path.join(left_eye_folder, left_iris_status, f"{username}_left_" + Path(image_path).name)
        right_eye_filename = os.path.join(right_eye_folder, right_iris_status, f"{username}_right_" + Path(image_path).name)

        cv2.imwrite(left_eye_filename, left_eye)
        cv2.imwrite(right_eye_filename, right_eye)

        # 눈 이미지 상하 배치 및 저장
        combined_eye_image = np.vstack([left_eye, right_eye])
        combined_eye_filename = os.path.join(combined_eye_folder, f"{username}_" + Path(image_path).name)
        cv2.imwrite(combined_eye_filename, combined_eye_image)

        # 로그 파일에 기록
        with open(os.path.join(output_folder, "log.txt"), "a") as log_file:
            log_file.write(f"{Path(image_path).name}: Left Eye: {left_iris_status}, Right Eye: {right_iris_status}\n")


if __name__ == "__main__":
    files = list(Path(base_input_folder).rglob("*.jpg"))
    for file_path in files:
        username = os.path.basename(os.path.dirname(file_path))
        process_image(file_path, base_output_folder, username)
    print("Processing completed.")
