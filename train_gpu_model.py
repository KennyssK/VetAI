import pandas as pd
import numpy as np
import joblib
import gc
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- НАСТРОЙКА GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- КОНФИГУРАЦИЯ v15.0 ---
DATASET_PATH = 'simulated_data_1500k_v1.3.csv'
MODEL_PATH = 'full_neural_network_model_v15.h5'
PREPROCESSOR_PATH = 'full_preprocessor_v15.pkl'
ENCODER_PATH = 'full_label_encoder_v15.pkl'
FEATURES_PATH = 'full_feature_names_v15.pkl'  # <--- НОВЫЙ ФАЙЛ

BATCH_SIZE = 1024
EPOCHS = 100

# 1. Загрузка данных
print("1. Загрузка данных...")
df = pd.read_csv(DATASET_PATH)
feature_names = [col for col in df.columns if col != 'Label']

# --- СОХРАНЯЕМ ИМЕНА ПРИЗНАКОВ СРАЗУ ---
print(f"Сохранение имен признаков ({len(feature_names)} шт.) в {FEATURES_PATH}...")
joblib.dump(feature_names, FEATURES_PATH)

# 2. Масштабирование
print("2. Подготовка признаков...")
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_names]).astype(np.float32)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Label'])
num_classes = len(label_encoder.classes_)

del df
gc.collect()

# 3. Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

del X, y
gc.collect()

# 4. Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(50000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 5. Архитектура
def build_robust_model(input_dim, num_classes):
    reg = keras.regularizers.l2(1e-4) 
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(1024, kernel_initializer='he_normal', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, kernel_initializer='he_normal', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=reg),
        keras.layers.Activation('swish'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_robust_model(len(feature_names), num_classes)

# 6. Обучение
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

print("\n--- Запуск Robust обучения (v15.0) ---")
model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[lr_reducer, early_stop])

# 7. Сохранение артефактов
print("\nСохранение модели и препроцессоров...")
model.save(MODEL_PATH)
joblib.dump(scaler, PREPROCESSOR_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

_, acc = model.evaluate(test_dataset, verbose=0)
print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО. ИТОГОВАЯ ТОЧНОСТЬ: {acc:.4f}")
print(f"Все файлы (model, scaler, encoder, feature_names) v15 сохранены.")