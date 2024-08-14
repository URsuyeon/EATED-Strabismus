# cut_fin.py 눈 검출 및 크롭 도구
이 스크립트는 얼굴 이미지를 처리하여 왼쪽 및 오른쪽 눈을 검출하고, 크롭하여 저장하는 기능을 제공합니다. <br/> 
Mediapipe 라이브러리를 사용하여 얼굴 랜드마크를 탐지하고, 얼굴을 크롭하고 눈을 검출하며, 동공의 위치를 분석하는 작업을 수행합니다. <br/> 
<br/> <br/> <br/> 

## 주요 기능
입력 이미지에서 얼굴을 검출하고 얼굴 영역을 크롭.
눈의 위치를 탐지하고, 설정된 크기로 눈 이미지를 리사이즈.
왼쪽 및 오른쪽 눈 이미지를 결합하여 출력.
동공 위치를 분석하여 시선 방향(왼쪽, 오른쪽, 위, 아래, 중앙)을 결정.
처리된 눈 이미지 및 동공 위치 정보를 저장.
<br/> <br/> 

## 요구 사항
스크립트는 다음의 Python 라이브러리를 필요로 합니다:

OpenCV (cv2) <br/> 
Mediapipe (mediapipe) <br/> 
Numpy (numpy) <br/> 
PIL (Python Imaging Library) <br/> 
<br/> <br/> 

## 입력 및 출력 구조
입력 폴더: 기본 입력 폴더에는 사용자별로 서브 폴더가 존재하며, 각 서브 폴더에는 사용자의 얼굴 이미지 파일들(PNG, JPG, JPEG 형식)이 위치합니다.<br/> 
출력 폴더: 크롭된 눈 이미지, 결합된 눈 이미지, 동공 상태 정보가 저장된 텍스트 파일이 해당 사용자 이름의 출력 폴더에 저장됩니다.
<br/> <br/> 

## 주요 상수
PADDING: 눈 영역 주변에 추가할 여백의 크기.<br/> 
FACE_SIZE: 얼굴 크롭 후 리사이즈할 이미지 크기 (1024x1024).<br/> 
FINAL_SIZE: 최종 눈 이미지의 크기 (32x32).<br/> 
threshold_X / threshold_Y: 동공 위치에 따라 시선 방향을 결정할 임계값.<br/> 
<br/> 

## 함수 설명
1. crop_face(image, detection)
얼굴 검출 결과를 바탕으로 입력 이미지에서 얼굴을 크롭합니다.

2. detect_eyes(face_image)
제공된 얼굴 이미지에서 눈 랜드마크를 탐지합니다.

3. extract_eye_image(image, eye_points)
주어진 눈 랜드마크를 기준으로 눈 이미지를 추출하고 리사이즈합니다.

4. combine_eyes(left_eye, right_eye)
왼쪽 눈과 오른쪽 눈 이미지를 결합한 후 리사이즈합니다.

5. calculate_eye_center(face_landmarks, eye_indices, image_shape)
얼굴 랜드마크를 기반으로 눈의 중심 좌표를 계산합니다.

6. determine_iris_position(eye_center, iris_center)
눈 중심과 동공 위치를 기준으로 시선 방향을 결정합니다.
<br/> <br/> <br/> 

## 실행 방법
기본 입력 폴더에 사용자별로 얼굴 이미지 파일을 준비합니다.<br/> 
스크립트를 실행하면 각 사용자의 이미지에 대해 얼굴과 눈을 검출하고, 처리된 결과가 출력 폴더에 저장됩니다.<br/> 
처리된 이미지와 동공 정보가 각각의 출력 폴더에 저장됩니다.
<br/> <br/> 


### 실행 예시
스크립트가 실행되면 각 사용자 폴더에 대한 이미지 처리가 완료되며, 출력 경로에 다음 파일들이 저장됩니다:<br/> <br/> 

결합된 눈 이미지: 사용자의 왼쪽 및 오른쪽 눈 이미지를 결합한 이미지.<br/> 
왼쪽 및 오른쪽 눈 이미지: 각각 크롭된 눈 이미지.<br/> 
동공 상태 정보 파일: 왼쪽과 오른쪽 눈의 동공 위치를 기반으로 한 시선 방향 정보.<br/> 
<br/> 

# Mediapipe를 사용한 눈과 눈동자 랜드마크 검출
이 코드는 MediaPipe Face Mesh를 사용하여 얼굴 랜드마크 중 특히 눈과 눈동자의 랜드마크를 검출하고 시각화하는 기능을 제공합니다.<br/> 
입력 이미지를 처리한 후 눈과 눈동자에 대한 랜드마크를 검출하고, 이를 이미지에 표시하여 시각화합니다.<br/> 
<br/> 
## 주요 기능
눈 랜드마크 검출: 좌우 눈의 랜드마크를 검출하고 이미지를 통해 시각화합니다.<br/> 
눈동자 랜드마크 검출: 좌우 눈동자의 랜드마크 및 눈동자의 중심 좌표를 검출합니다.<br/> 
이미지 시각화: 검출된 눈과 눈동자의 랜드마크를 원본 이미지에 그려줍니다.<br/> 
<br/> 

## 코드 실행
이미지를 분석하고 결과를 시각화하려면, analyze_image_with_landmarks와 display_results 함수를 실행합니다. 아래의 샘플 코드에서 image_path에 사용할 이미지 경로를 설정하세요.

```
#이미지 경로 설정
image_path = r'C:\Users\user\PycharmProjects\eated\data\eyes\Jye\20240806_213852(0).jpg'
  
#이미지 분석 및 시각화
original_image, annotated_image = analyze_image_with_landmarks(image_path)
display_results(original_image, annotated_image)
```
<br/> <br/> 

## 코드 설명
analyze_image_with_landmarks(image_path): 이 함수는 주어진 이미지에서 눈과 눈동자의 랜드마크를 검출하고, 랜드마크가 표시된 이미지를 반환합니다. 검출된 좌표는 cv2.circle과 cv2.putText로 표시됩니다.<br/> 

눈 랜드마크: 좌우 눈의 랜드마크는 각각 녹색으로 표시되며, 인덱스 번호도 함께 표시됩니다.<br/> 
눈동자 랜드마크: 눈동자 주변의 랜드마크는 노란색으로, 눈동자의 중심은 빨간색으로 표시됩니다.<br/> 
display_results(original_image, annotated_image): 이 함수는 원본 이미지와 랜드마크가 표시된 이미지를 나란히 출력합니다. Matplotlib을 사용하여 결과를 시각적으로 비교할 수 있습니다.<br/> 
<br/> 

## 구조
이미지 파일: image_path로 입력한 이미지를 처리합니다.<br/> 
Mediapipe Face Mesh: 얼굴의 468개 랜드마크 중 눈과 눈동자에 관련된 좌표를 추출합니다.<br/> 
OpenCV: 눈과 눈동자 랜드마크를 이미지 위에 그립니다.<br/> 
Matplotlib: 처리된 이미지를 시각화합니다.<br/> 
<br/> 

## 사용 예시
이미지 경로를 설정합니다.<br/> 
analyze_image_with_landmarks 함수를 통해 눈과 눈동자 랜드마크를 검출합니다.<br/> 
display_results 함수로 검출 결과를 시각화합니다.<br/> 
<br/> 

### 예시 결과
원본 이미지와 랜드마크가 표시된 이미지를 나란히 비교하여 눈과 눈동자의 위치를 확인할 수 있습니다.<br/> 

## 주의사항
이미지 크기: 입력 이미지의 크기와 비율에 따라 검출된 좌표가 달라질 수 있습니다. 다양한 해상도의 이미지를 테스트해 보세요.<br/> 
단일 얼굴 검출: 이 프로젝트는 한 이미지에서 최대 1개의 얼굴만 처리하도록 설정되어 있습니다.<br/> 
