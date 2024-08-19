import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# 데이터 경로 설정
train_dir = 'C:/Users/user/Desktop/data/AI/Train'
val_dir = 'C:/Users/user/Desktop/data/AI/Validation'
test_dir = 'C:/Users/user/Desktop/data/AI/Test'

# 이미지 로드 및 전처리 함수
def load_images_from_folder(folder, image_size=(32, 32)):
    images = []
    labels = []
    for label, category in enumerate(['Abnormal', 'Normal']):
        category_folder = os.path.join(folder, category)
        for filename in os.listdir(category_folder):
            img_path = os.path.join(category_folder, filename)
            img = Image.open(img_path).resize(image_size)
            img = np.array(img) / 255.0  # 정규화
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# 데이터 로드
train_images, train_labels = load_images_from_folder(train_dir)
val_images, val_labels = load_images_from_folder(val_dir)
test_images, test_labels = load_images_from_folder(test_dir)

# 데이터셋 분리
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# 데이터셋 배치 및 셔플
train_ds = train_ds.shuffle(buffer_size=len(train_images)).batch(32)
val_ds = val_ds.batch(32)
test_ds = test_ds.batch(32)

# ResNet50 모델 불러오기
resnet_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))

# 새로운 모델 생성
inputs = tf.keras.Input(shape=(32, 32, 3))
x = resnet_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# 모델 학습
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds
)

# 모델 저장
model.save('C:/Users/user/Desktop/data/AI/my_model.keras')  # .keras 확장자 사용

# 모델 로드
loaded_model = tf.keras.models.load_model('C:/Users/user/Desktop/data/AI/my_model.keras')

# 테스트 데이터로 평가
test_loss, test_accuracy, test_auc = loaded_model.evaluate(test_ds)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test AUC: {test_auc}')

# 예측
predictions = loaded_model.predict(test_ds)
predictions = (predictions > 0.5).astype(int)  # Binary classification

# 실제 라벨과 예측 라벨을 비교
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
accuracy = np.mean(true_labels == predictions)
auc = tf.keras.metrics.AUC()(true_labels, predictions).numpy()

print(f'Accuracy: {accuracy}')
print(f'AUC: {auc}')

# 레이어 정보 출력
for layer in model.layers:
    print(layer.name, layer.output_shape)
