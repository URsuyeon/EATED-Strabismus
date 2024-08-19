import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score

# 데이터 경로 설정
train_dir = 'C:/Users/user/Desktop/data/AI/Train'
val_dir = 'C:/Users/user/Desktop/data/AI/Validation'
test_dir = 'C:/Users/user/Desktop/data/AI/Test'

# 이미지 전처리 및 데이터 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  # 이미지 크기 조정
    batch_size=32,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(32, 32),  # 이미지 크기 조정
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),  # 이미지 크기 조정
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

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

# 모델 체크포인트 설정
checkpoint = ModelCheckpoint(
    'best_model.keras',  # 파일 확장자를 .keras로 변경
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

# 테스트 데이터로 평가
test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test AUC: {test_auc}')

# 예측
predictions = model.predict(test_generator)
predictions = (predictions > 0.5).astype(int)  # Binary classification

# 실제 라벨과 예측 라벨을 비교
true_labels = test_generator.classes
accuracy = accuracy_score(true_labels, predictions)
auc = roc_auc_score(true_labels, predictions)

print(f'Accuracy: {accuracy}')
print(f'AUC: {auc}')

# 레이어 정보 출력
for layer in model.layers:
    print(layer.name, layer.output_shape)
