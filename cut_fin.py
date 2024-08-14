import cv2
import mediapipe as mp
import os
from pathlib import Path
import numpy as np
from PIL import Image

# 상수 정의
PADDING = 30
FACE_SIZE = (1048, 1048)
FINAL_SIZE = (256, 256)
threshold = 0.35

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 입력 폴더와 출력 폴더의 기본 경로
base_input_folder = r'C:\Users\user\PycharmProjects\eated\data\eyes'
base_output_folder = r'C:\Users\user\PycharmProjects\eated\data\cropped'

# 눈의 랜드마크 인덱스 정의
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

# 눈동자 중심 랜드마크 인덱스 정의
LEFT_IRIS_CENTER_INDEX = 468
RIGHT_IRIS_CENTER_INDEX = 473


def crop_face(image, detection):
    ih, iw, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    x_min = int(max(bboxC.xmin * iw, 0))
    y_min = int(max(bboxC.ymin * ih, 0))
    x_max = int(min((bboxC.xmin + bboxC.width) * iw, iw))
    y_max = int(min((bboxC.ymin + bboxC.height) * ih, ih))

    if x_min >= x_max or y_min >= y_max:
        return None

    cropped_face = image[y_min:y_max, x_min:x_max]
    if cropped_face.size == 0:
        return None

    return cv2.resize(cropped_face, FACE_SIZE)


def detect_eyes(face_image):
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
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_image = image[max(y - PADDING, 0):min(y + h + PADDING, image.shape[0]),
                      max(x - PADDING, 0):min(x + w + PADDING, image.shape[1])]
    return cv2.resize(eye_image, FINAL_SIZE)


def combine_eyes(left_eye, right_eye):
    combined_height = left_eye.shape[0] + right_eye.shape[0]
    combined_width = max(left_eye.shape[1], right_eye.shape[1])
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_image[:left_eye.shape[0], :left_eye.shape[1]] = left_eye
    combined_image[left_eye.shape[0]:, :right_eye.shape[1]] = right_eye
    return cv2.resize(combined_image, FINAL_SIZE)


def calculate_eye_center(face_landmarks, eye_indices, image_shape):
    eye_landmarks = [face_landmarks.landmark[idx] for idx in eye_indices]
    x = sum([lm.x for lm in eye_landmarks]) / len(eye_landmarks) * image_shape[1]
    y = sum([lm.y for lm in eye_landmarks]) / len(eye_landmarks) * image_shape[0]
    return int(x), int(y)


def determine_iris_position(eye_center, iris_center):
    x_ratio = (iris_center[0] - eye_center[0]) / (FINAL_SIZE[0] / 2)
    y_ratio = (iris_center[1] - eye_center[1]) / (FINAL_SIZE[1] / 2)


    if abs(x_ratio) > (0.5-threshold):
        return 'Left' if x_ratio < 0 else 'Right'
    if abs(y_ratio) > (0.4-threshold):
        return 'Up' if y_ratio < 0 else 'Down'
    return 'Center'


def detect_and_crop_eyes(face_image):
    face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    resized_image_pil = face_image_pil.resize(FACE_SIZE)
    resized_image = cv2.cvtColor(np.array(resized_image_pil), cv2.COLOR_RGB2BGR)

    results = face_mesh.process(resized_image)
    if not results.multi_face_landmarks:
        return None, None, None

    face_landmarks = results.multi_face_landmarks[0]

    # Calculate eye centers
    left_eye_center = calculate_eye_center(face_landmarks, LEFT_EYE_INDICES, resized_image.shape)
    right_eye_center = calculate_eye_center(face_landmarks, RIGHT_EYE_INDICES, resized_image.shape)

    # Calculate iris centers
    left_iris_center = (int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                        int(face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))
    right_iris_center = (int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].x * resized_image.shape[1]),
                         int(face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX].y * resized_image.shape[0]))

    # Determine eye direction
    left_iris_status = determine_iris_position(left_eye_center, left_iris_center)
    right_iris_status = determine_iris_position(right_eye_center, right_iris_center)

    iris_status = f"Left: {left_iris_status}, Right: {right_iris_status}"

    # Extract and resize eye images
    def extract_and_resize_eye_image(eye_indices):
        eye_points = [(int(face_landmarks.landmark[idx].x * resized_image.shape[1]),
                       int(face_landmarks.landmark[idx].y * resized_image.shape[0])) for idx in eye_indices]
        return extract_eye_image(resized_image, np.array(eye_points))

    left_eye_image = extract_and_resize_eye_image(LEFT_EYE_INDICES)
    right_eye_image = extract_and_resize_eye_image(RIGHT_EYE_INDICES)

    return left_eye_image, right_eye_image, iris_status


def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    # Detect face and crop
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

    # Detect and crop eyes
    left_eye, right_eye, iris_status = detect_and_crop_eyes(cropped_face)
    if left_eye is None or right_eye is None:
        print(f"Failed to detect eyes for {image_path}")
        return

    # Combine eyes and save results
    final_image = combine_eyes(left_eye, right_eye)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    final_image_path = os.path.join(output_folder, f"{name}_combined{ext}")
    cv2.imwrite(final_image_path, final_image)

    # Save eye images and information
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
    Path(output_folder).mkdir(parents=True, exist_ok=True)

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
