import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# --- GPT 변환 (그대로 사용) ---
def gpt_transform(signal, m=50, n=100):
    if np.max(signal) == np.min(signal):
        return np.zeros((m, n))
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    T = len(signal)
    gp_image = np.zeros((m, n))
    for t in range(T):
        v = signal[t]
        i = int((1 - v) * m)
        j = int(t / T * n)
        i = min(m - 1, max(0, i))
        j = min(n - 1, max(0, j))
        gp_image[i, j] += 1
    return gp_image

# --- 증강 함수들 (그대로 사용) ---
def augment_noise(signal, std=0.002):
    return signal + np.random.normal(0, std, size=signal.shape)

def augment_shift(signal, shift_max=10):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift)

def augment_scale(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

# --- 회귀용 데이터셋 생성 ---
def make_dataset_with_coordinates(csv_path, segment_len=300, augment_factor=3):
    df = pd.read_csv(csv_path)
    sensor_columns = [col for col in df.columns if col.startswith('sensor')]
    num_samples = df.shape[0] // segment_len

    X, y = [], []

    for i in range(num_samples):
        segment = df.iloc[i*segment_len:(i+1)*segment_len]
        # 위치 좌표로부터 레이블 구성 (예: x, y 열 존재한다고 가정)
        x_coord = float(segment['x'].iloc[0])
        y_coord = float(segment['y'].iloc[0])
        sensor_signals = [segment[ch].values for ch in sensor_columns]
        variants_per_sensor = []

        for sig in sensor_signals:
            variants = [sig]
            variants.append(augment_noise(sig))
            variants.append(augment_shift(sig))
            variants.append(augment_scale(sig))
            variants_per_sensor.append(variants)

        for j in range(augment_factor + 1):
            stacked = []
            for sensor_variants in variants_per_sensor:
                gpt_img = gpt_transform(sensor_variants[j])
                stacked.append(gpt_img)
            gpf_image = np.stack(stacked, axis=-1)
            X.append(gpf_image)
            y.append([x_coord, y_coord])  # 좌표 레이블

    return np.array(X), np.array(y)

# --- 실행 ---
X, y = make_dataset_with_coordinates(
    csv_path="C:/Users/psw09/Desktop/pvdf test/carbon619_regression.csv",
    segment_len=300,
    augment_factor=3
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 회귀용 모델 정의 ---
model = models.Sequential([
    layers.Input(shape=(50, 100, X.shape[-1])),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
    layers.Dense(2)  # 좌표 회귀 출력 (x, y)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# --- 학습 ---
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# --- 평가 ---
loss, mae = model.evaluate(X_val, y_val)
print(f"✅ 평균 위치 오차 (MAE): {mae:.2f} mm")

# --- 시각화 (그대로 사용 가능) ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("regression_training_curve.png", dpi=300)
plt.show()
