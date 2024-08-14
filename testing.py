import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 눈의 랜드마크 인덱스 정의
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

# 눈동자 랜드마크 인덱스 정의
LEFT_IRIS_INDICES = [469, 470, 471, 472]
RIGHT_IRIS_INDICES = [474, 475, 476, 477]

# 눈동자 중심 랜드마크 인덱스 정의
LEFT_IRIS_CENTER_INDEX = 468
RIGHT_IRIS_CENTER_INDEX = 473

def analyze_image_with_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe
    results = face_mesh.process(rgb_image)

    annotated_image = image.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            # Draw left eye landmarks
            for idx in LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)  # Green for eye landmarks
                cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw right eye landmarks
            for idx in RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)  # Green for eye landmarks
                cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw left iris landmarks
            for idx in LEFT_IRIS_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 255), -1)  # Yellow for iris landmarks
                cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw right iris landmarks
            for idx in RIGHT_IRIS_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 255), -1)  # Yellow for iris landmarks
                cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw left iris center
            left_iris_center = face_landmarks.landmark[LEFT_IRIS_CENTER_INDEX]
            left_center_x = int(left_iris_center.x * w)
            left_center_y = int(left_iris_center.y * h)
            cv2.circle(annotated_image, (left_center_x, left_center_y), 3, (255, 0, 0), -1)  # Red for iris center
            cv2.putText(annotated_image, str(LEFT_IRIS_CENTER_INDEX), (left_center_x, left_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw right iris center
            right_iris_center = face_landmarks.landmark[RIGHT_IRIS_CENTER_INDEX]
            right_center_x = int(right_iris_center.x * w)
            right_center_y = int(right_iris_center.y * h)
            cv2.circle(annotated_image, (right_center_x, right_center_y), 3, (255, 0, 0), -1)  # Red for iris center
            cv2.putText(annotated_image, str(RIGHT_IRIS_CENTER_INDEX), (right_center_x, right_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    return rgb_image, annotated_image

def display_results(original_image, annotated_image):
    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Display the original image with axes
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('on')  # Show axes
    ax[0].set_xlabel('X-axis')
    ax[0].set_ylabel('Y-axis')

    # Display the annotated image with landmarks and axes
    ax[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Annotated Image with Eye and Iris Landmarks')
    ax[1].axis('on')  # Show axes
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Y-axis')

    plt.show()

# Path to your image
image_path = r'C:\Users\user\PycharmProjects\eated\data\eyes\Jye\20240806_213852(0).jpg'

# Analyze the image with landmarks
original_image, annotated_image = analyze_image_with_landmarks(image_path)

# Display the results
display_results(original_image, annotated_image)
