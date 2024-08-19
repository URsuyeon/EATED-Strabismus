import os
import shutil
import random
from tqdm import tqdm

def create_directories(base_dir):
    """학습, 검증, 테스트 데이터셋을 위한 디렉토리 생성"""
    for split in ['Train', 'Validation', 'Test']:
        for label in ['Normal', 'Abnormal']:
            os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)

def is_image_file(filename):
    """파일이 이미지 파일인지 확인"""
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS

def split_data(source_dir, dest_dir, label, train_split=0.64, val_split=0.16):
    """데이터를 학습, 검증, 테스트 세트로 나누기"""
    src_path = source_dir
    train_path = os.path.join(dest_dir, 'Train', label)
    val_path = os.path.join(dest_dir, 'Validation', label)
    test_path = os.path.join(dest_dir, 'Test', label)

    # 원본 디렉토리의 이미지 파일 리스트 가져오기
    files = [f for f in os.listdir(src_path) if is_image_file(f)]
    random.shuffle(files)  # 파일을 섞어서 랜덤하게 분할

    # 분할 인덱스 계산
    total_files = len(files)
    train_end = int(total_files * train_split)
    val_end = train_end + int(total_files * val_split)

    # 학습, 검증, 테스트 파일로 분할
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # 각 세트에 파일 복사
    for file in tqdm(train_files, desc=f'{label} 학습 파일 복사'):
        shutil.copy(os.path.join(src_path, file), os.path.join(train_path, file))
    for file in tqdm(val_files, desc=f'{label} 검증 파일 복사'):
        shutil.copy(os.path.join(src_path, file), os.path.join(val_path, file))
    for file in tqdm(test_files, desc=f'{label} 테스트 파일 복사'):
        shutil.copy(os.path.join(src_path, file), os.path.join(test_path, file))

def main():
    # 원본 데이터 디렉토리 설정
    normal_src = r'C:\Users\user\Desktop\data\cropped\combined_eye'
    strabismus_src = r'C:\Users\user\Desktop\data\cropped\strabismus'

    # 목적지 디렉토리 설정
    dest_dir = r'C:\Users\user\Desktop\data\AI'

    # 목적지 디렉토리가 존재하지 않으면 상위 디렉토리와 함께 생성
    if not os.path.exists(dest_dir):
        parent_dir = os.path.dirname(dest_dir)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)  # 상위 디렉토리 생성
        os.makedirs(dest_dir)  # 목적지 디렉토리 생성

    # 목적지 디렉토리 내의 하위 디렉토리 생성
    create_directories(dest_dir)

    # 정상 데이터와 사시 데이터를 학습, 검증, 테스트 세트로 분할 및 복사
    split_data(normal_src, dest_dir, 'Normal')
    split_data(strabismus_src, dest_dir, 'Abnormal')

if __name__ == "__main__":
    main()
