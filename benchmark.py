import numpy as np
import tensorflow as tf
import time
import joblib

# Настройки
MODEL_PATH = 'full_neural_network_model_v15_opt.h5'
FEATURES_PATH = 'full_feature_names_v15.pkl'
BATCH_SIZE = 1024  # Размер пачки для теста

# 1. Загрузка данных и модели
print(f"--- Загрузка модели {MODEL_PATH} ---")
# Загружаем без компиляции, так как нам нужно только предсказание
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
features = joblib.dump = joblib.load(FEATURES_PATH)
num_features = len(features)

# 2. Подготовка фейковых данных (имитация 1024 пациентов)
print(f"--- Подготовка данных для {BATCH_SIZE} пациентов ---")
fake_data = np.random.rand(BATCH_SIZE, num_features).astype(np.float32)

# 3. Разогрев (Warm-up) - обязателен для GPU!
print("--- Разогрев GPU (первый прогон) ---")
model.predict(fake_data[:1], verbose=0)

# 4. Основной тест
print(f"--- Запуск теста (Batch Size: {BATCH_SIZE}) ---")
start_time = time.time()
model.predict(fake_data, batch_size=BATCH_SIZE, verbose=0)
end_time = time.time()

# 5. Расчеты
total_ms = (end_time - start_time) * 1000
per_patient_ms = total_ms / BATCH_SIZE

print("\n" + "="*30)
print(f"РЕЗУЛЬТАТЫ BENCHMARK (RTX 3060 + Mixed Precision)")
print(f"Общее время пачки: {total_ms:.2f} ms")
print(f"Время на 1 пациента: {per_patient_ms:.4f} ms")
print("="*30)