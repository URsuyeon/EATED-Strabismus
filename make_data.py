import cv2
import os
import numpy as np
from pathlib import Path
import uuid

# 입력 폴더와 출력 폴더의 기본 경로
base_cropped_folder = r'C:\Users\user\Desktop\data\cropped'
strabismus_folder = os.path.join(base_cropped_folder, "strabismus")
os.makedirs(strabismus_folder, exist_ok=True)

# 사시 유형을 정의
strabismus_types = {
    "Eso_": [("Right", "Center"), ("Center", "Left")],  # 내사시
    "Exo_": [("Left", "Center"), ("Center", "Right")],  # 외사시
    "Hyper_": [("Up", "Center"), ("Center", "Up")],     # 상사시
    "Hypo_": [("Down", "Center"), ("Center", "Down")]   # 하사시
}

# UUID 기록 파일 경로
uuid_log_file = os.path.join(strabismus_folder, "generated_uuids.txt")
# 사용된 이미지 기록 파일 경로
used_images_log_file = os.path.join(strabismus_folder, "used_images_log.txt")

def get_eye_image(eye_folder, direction, username):
    """특정 사용자명과 방향에 맞는 눈 이미지를 불러옴"""
    direction_folder = os.path.join(eye_folder, direction)
    user_files = [f for f in Path(direction_folder).glob(f"{username}_*.jpg")]
    if not user_files:
        return None
    return str(user_files[0])  # 첫 번째 파일명을 반환

def read_file_lines(file_path):
    """파일에서 줄을 읽어 리스트로 반환"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def write_file_lines(file_path, lines):
    """리스트를 파일에 줄 단위로 작성"""
    with open(file_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

def generate_unique_uuid():
    """고유한 UUID를 생성하고, 이미 생성된 UUID와 충돌하는지 확인"""
    while True:
        unique_id = str(uuid.uuid4())
        if not is_uuid_generated(unique_id):
            return unique_id

def is_uuid_generated(unique_id):
    """UUID가 이미 생성된 UUID 목록에 존재하는지 확인"""
    generated_uuids = read_file_lines(uuid_log_file)
    return unique_id in generated_uuids

def log_generated_uuid(unique_id):
    """생성된 UUID를 로그 파일에 기록"""
    existing_uuids = read_file_lines(uuid_log_file)
    existing_uuids.append(unique_id)
    write_file_lines(uuid_log_file, existing_uuids)

def log_used_images(left_image_path, right_image_path):
    """사용된 이미지 파일명을 로그 파일에 기록"""
    with open(used_images_log_file, 'a') as f:
        f.write(f"{left_image_path},{right_image_path}\n")

def create_strabismus_data(username, strabismus_type, left_direction, right_direction):
    """사시 데이터를 생성하여 저장하는 함수"""
    left_eye_folder = os.path.join(base_cropped_folder, "left_eye")
    right_eye_folder = os.path.join(base_cropped_folder, "right_eye")

    left_center_image_path = get_eye_image(left_eye_folder, 'Center', username)
    right_center_image_path = get_eye_image(right_eye_folder, 'Center', username)

    if left_center_image_path is None or right_center_image_path is None:
        print(f"Center eye images missing for {username}")
        return

    left_eye_image_path = get_eye_image(left_eye_folder, left_direction, username)
    right_eye_image_path = get_eye_image(right_eye_folder, right_direction, username)

    if left_eye_image_path is None or right_eye_image_path is None:
        print(f"Eye images missing for {username} in directions {left_direction} or {right_direction}")
        return

    left_eye_image = cv2.imread(left_eye_image_path)
    right_eye_image = cv2.imread(right_eye_image_path)
    left_center_image = cv2.imread(left_center_image_path)
    right_center_image = cv2.imread(right_center_image_path)

    # 눈 이미지 상하로 결합
    combined_image_left_random_right_center = np.vstack([left_eye_image, right_center_image])
    combined_image_left_center_right_random = np.vstack([left_center_image, right_eye_image])

    # 고유한 UUID 추가
    unique_id = generate_unique_uuid()
    log_generated_uuid(unique_id)
    log_used_images(left_eye_image_path, right_eye_image_path)

    # 저장할 파일명에 사시 유형 반영
    filename_left_random_right_center = f"{username}_{strabismus_type}left_{left_direction}_right_center_{unique_id}.jpg"
    filename_left_center_right_random = f"{username}_{strabismus_type}left_center_right_{right_direction}_{unique_id}.jpg"

    output_path_left_random_right_center = os.path.join(strabismus_folder, filename_left_random_right_center)
    output_path_left_center_right_random = os.path.join(strabismus_folder, filename_left_center_right_random)

    # 이미지 저장
    cv2.imwrite(output_path_left_random_right_center, combined_image_left_random_right_center)
    cv2.imwrite(output_path_left_center_right_random, combined_image_left_center_right_random)

    print(f"Strabismus data created for {username}: {filename_left_random_right_center}, {filename_left_center_right_random}")

def get_all_combinations(usernames):
    """모든 조합을 생성하여 반환"""
    combinations = []
    for username in usernames:
        for strabismus_type, directions in strabismus_types.items():
            for left_direction, right_direction in directions:
                if left_direction == 'Center':
                    for right_dir in ['Left', 'Right', 'Up', 'Down']:
                        combinations.append((username, strabismus_type, left_direction, right_dir))
                elif right_direction == 'Center':
                    for left_dir in ['Left', 'Right', 'Up', 'Down']:
                        combinations.append((username, strabismus_type, left_dir, right_direction))
    return combinations

def generate_random_strabismus_data():
    """모든 사용자에 대해 중복 없는 데이터를 생성"""
    # 사용자명을 가져오는 로직
    left_eye_user_folders = Path(os.path.join(base_cropped_folder, "left_eye", "Center"))
    right_eye_user_folders = Path(os.path.join(base_cropped_folder, "right_eye", "Center"))

    # 공통된 사용자명 추출
    left_usernames = {f.stem.split('_')[0] for f in left_eye_user_folders.glob("*.jpg")}
    right_usernames = {f.stem.split('_')[0] for f in right_eye_user_folders.glob("*.jpg")}
    usernames = left_usernames.intersection(right_usernames)  # 두 목록에 공통된 사용자만 처리

    # 모든 조합 생성
    combinations = get_all_combinations(usernames)

    # 중복된 조합을 방지하기 위한 세트
    generated_combinations = set()

    # 생성할 데이터 조합을 반복
    for username, strabismus_type, left_direction, right_direction in combinations:
        # 조합이 이미 생성된 조합 세트에 있는지 확인
        if (username, strabismus_type, left_direction, right_direction) not in generated_combinations:
            create_strabismus_data(username, strabismus_type, left_direction, right_direction)
            generated_combinations.add((username, strabismus_type, left_direction, right_direction))

    print("Random strabismus data generation completed.")

if __name__ == "__main__":
    # 중복 없는 최대 데이터를 생성
    generate_random_strabismus_data()
