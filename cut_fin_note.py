import cv2
import mediapipe as mp
import os
from pathlib import Path
import numpy as np
from PIL import Image

# 상수 정의
PADDING = 30  # 눈 이미지의 가장자리에 추가할 패딩 크기
FACE_SIZE = (1024,1024)  # 얼굴 이미지를 크롭한 후 조정할 크기
FINAL_SIZE = (32, 32)  # 최종 눈 이미지의 크기
threshold_X = 0.36  # 시선 방향 판단의 X임계값
threshold_Y = 0.46  # 시선 방향 판단의 Y임계값

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # 정적인 이미지 모드
    max_num_faces=1,         # 최대 얼굴 수
    refine_landmarks=True,   # 랜드마크 정제 여부
    min_detection_confidence=0.5  # 최소 검출 신뢰도
)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 입력 폴더와 출력 폴더의 기본 경로
base_input_folder = r'C:\Users\user\Desktop\data\eyes'
base_output_folder = r'C:\Users\user\Desktop\data\cropped'

# 눈의 랜드마크 인덱스 정의
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

# 눈동자 중심 랜드마크 인덱스 정의
LEFT_IRIS_CENTER_INDEX = 468
RIGHT_IRIS_CENTER_INDEX = 473


def crop_face(image, detection):
    """
    얼굴 이미지에서 얼굴 부분을 크롭합니다.
    :param image: 원본 이미지
    :param detection: 얼굴 검출 결과
    :return: 크롭된 얼굴 이미지 또는 None
    """
    ih, iw, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    x_min = int(max(bboxC.xmin * iw, 0))  # 얼굴의 x 시작 좌표
    y_min = int(max(bboxC.ymin * ih, 0))  # 얼굴의 y 시작 좌표
    x_max = int(min((bboxC.xmin + bboxC.width) * iw, iw))  # 얼굴의 x 끝 좌표
    y_max = int(min((bboxC.ymin + bboxC.height) * ih, ih))  # 얼굴의 y 끝 좌표

    if x_min >= x_max or y_min >= y_max:
        return None

    cropped_face = image[y_min:y_max, x_min:x_max]  # 얼굴 부분 크롭
    if cropped_face.size == 0:
        return None

    return cv2.resize(cropped_face, FACE_SIZE)  # 얼굴 이미지 크기 조정


def detect_eyes(face_image):
    """
    얼굴 이미지에서 눈의 위치를 검출합니다.
    :param face_image: 얼굴 이미지
    :return: 왼쪽과 오른쪽 눈의 좌표 리스트
    """
    results = face_mesh.process(face_image)
    if not results.multi_face_landmarks:
        return None, None

    face_landmarks = results.multi_face_landmarks[0]
    left_eye_points = [(int(face_landmarks.landmark[idx].x * face_image.shape[1]),
                        int(face_landmarks.landmark[idx].y * face_image.shape[0])) for idx in LEFT_EYE_INDICES]
    right_eye_points = [(int(face_landmarks.landmark[idx].x * face_image.shape[1]),
                         int(face_landmarks.landmark[idx].y * face_image.shape[0])) for idx in RIGHT_EYE_INDICES]
    return np.array(left_eye_points), np.array(right_eye_points)


def extract_eye_image(image, eye_points):
    """
    주어진 눈 포인트를 사용하여 눈 이미지를 추출하고 크기를 조정합니다.
    :param image: 얼굴 이미지
    :param eye_points: 눈 랜드마크 포인트
    :return: 추출된 눈 이미지
    """
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_image = image[max(y - PADDING, 0):min(y + h + PADDING, image.shape[0]),
                      max(x - PADDING, 0):min(x + w + PADDING, image.shape[1])]
    return cv2.resize(eye_image, FINAL_SIZE)


def combine_eyes(left_eye, right_eye):
    """
    왼쪽과 오른쪽 눈 이미지를 결합합니다.
    :param left_eye: 왼쪽 눈 이미지
    :param right_eye: 오른쪽 눈 이미지
    :return: 결합된 눈 이미지
    """
    combined_height = left_eye.shape[0] + right_eye.shape[0]
    combined_width = max(left_eye.shape[1], right_eye.shape[1])
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_image[:left_eye.shape[0], :left_eye.shape[1]] = left_eye
    combined_image[left_eye.shape[0]:, :right_eye.shape[1]] = right_eye
    return cv2.resize(combined_image, FINAL_SIZE)


def calculate_eye_center(face_landmarks, eye_indices, image_shape):
    """
    눈의 중심 좌표를 계산합니다.
    :param face_landmarks: 얼굴 랜드마크
    :param eye_indices: 눈 랜드마크 인덱스
    :param image_shape: 이미지의 크기
    :return: 눈 중심 좌표 (x, y)
    """
    eye_landmarks = [face_landmarks.landmark[idx] for idx in eye_indices]
    x = sum([lm.x for lm in eye_landmarks]) / len(eye_landmarks) * image_shape[1]
    y = sum([lm.y for lm in eye_landmarks]) / len(eye_landmarks) * image_shape[0]
    return int(x), int(y)


def determine_iris_position(eye_center, iris_center):
    """
    눈동자의 위치에 따라 눈의 방향을 결정합니다.
    :param eye_center: 눈의 중심 좌표
    :param iris_center: 눈동자의 중심 좌표
    :return: 눈동자의 방향 ('Left', 'Right', 'Up', 'Down', 'Center')
    """
    x_ratio = (iris_center[0] - eye_center[0]) / (FINAL_SIZE[0] / 2)
    y_ratio = (iris_center[1] - eye_center[1]) / (FINAL_SIZE[1] / 2)

    if abs(x_ratio) > (0.5 - threshold_X):
        return 'Left' if x_ratio < 0 else 'Right'
    if abs(y_ratio) > (0.5 - threshold_Y):
        return 'Up' if y_ratio < 0 else 'Down'
    return 'Center'


def detect_and_crop_eyes(face_image):
    """
    얼굴 이미지에서 눈을 검출하고 크롭하여 반환합니다.
    :param face_image: 얼굴 이미지
    :return: 왼쪽과 오른쪽 눈 이미지, 눈동자 상태
    """
    face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    resized_image_pil = face_image_pil.resize(FACE_SIZE)
    resized_image = cv2.cvtColor(np.array(resized_image_pil), cv2.COLOR_RGB2BGR)

    results = face_mesh.process(resized_image)
    if not results.multi_face_landmarks:
        return None, None, None

    face_landmarks = results.multi_face_landmarks[0]

    # 눈의 중심 좌표 계산
    left_eye_center = calculate_eye_center(face_landmarks, LEFT_EYE_INDICES, resized_image.shape)
    right_eye_center = calculate_eye_center(face_landmarks, RIGHT_EYE_INDICES, resized_image.shape)

    # 눈동자의 중심 좌표 계산
    left_iris_center = (int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                        int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))
    right_iris_center = (int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                         int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))

    # 눈동자 상태 결정
    left_iris_status = determine_iris_position(left_eye_center, left_iris_center)
    right_iris_status = determine_iris_position(right_eye_center, right_iris_center)

    iris_status = f"Left: {left_iris_status}, Right: {right_iris_status}"

    # 눈 이미지 추출 및 크기 조정
    def extract_and_resize_eye_image(eye_indices):
        eye_points = [(int(face_landmarks.landmark[idx].x * resized_image.shape[1]),
                       int(face_landmarks.landmark[idx].y * resized_image.shape[0])) for idx in eye_indices]
        return extract_eye_image(resized_image, np.array(eye_points))

    left_eye_image = extract_and_resize_eye_image(LEFT_EYE_INDICES)
    right_eye_image = extract_and_resize_eye_image(RIGHT_EYE_INDICES)

    return left_eye_image, right_eye_image, iris_status


def process_image(image_path, output_folder):
    """
    이미지를 처리하고 결과를 저장합니다.
    :param image_path: 입력 이미지 경로
    :param output_folder: 출력 폴더 경로
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    # 얼굴 검출 및 크롭
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_results = face_detection.process(image_rgb)
        if not detection_results.detections:
            print(f"No face detected in {image_path}")
            return

        cropped_face = crop_face(image, detection_results.detections[0])
        if cropped_face is None:
            print(f"Failed to crop face for {image_path}")
            return

    # 눈 검출 및 크롭
    left_eye, right_eye, iris_status = detect_and_crop_eyes(cropped_face)
    if left_eye is None or right_eye is None:
        print(f"Failed to detect eyes for {image_path}")
        return

    # 눈 이미지 결합 및 결과 저장
    final_image = combine_eyes(left_eye, right_eye)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    final_image_path = os.path.join(output_folder, f"{name}_combined{ext}")
    cv2.imwrite(final_image_path, final_image)

    # 눈 이미지 및 정보 저장
    left_eye_path = os.path.join(output_folder, f"{name}_left_eye{ext}")
    right_eye_path = os.path.join(output_folder, f"{name}_right_eye{ext}")
    cv2.imwrite(left_eye_path, left_eye)
    cv2.imwrite(right_eye_path, right_eye)

    info_file = os.path.join(output_folder, f"{name}_eye_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Left Eye Iris Status: {iris_status.split(', ')[0].split(': ')[1]}\n")
        f.write(f"Right Eye Iris Status: {iris_status.split(', ')[1].split(': ')[1]}\n")

    print(f"Processed {image_path} and saved to {final_image_path}, {left_eye_path}, {right_eye_path}, and {info_file}")


def process_folder(input_folder, output_folder):
    """
    폴더 내의 모든 이미지를 처리합니다.
    :param input_folder: 입력 폴더 경로
    :param output_folder: 출력 폴더 경로
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성

    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path, output_folder)


# 입력 폴더 내의 서브 폴더를 사용자명으로 사용
user_folders = [f for f in os.listdir(base_input_folder) if os.path.isdir(os.path.join(base_input_folder, f))]

for username in user_folders:
    input_folder = os.path.join(base_input_folder, username)
    output_folder = os.path.join(base_output_folder, username)

    process_folder(input_folder, output_folder)

print("Processing completed.")