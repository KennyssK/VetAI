import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# 1. Загрузка компонентов
try:
    model = keras.models.load_model('neural_network_model.h5')
    preprocessor = joblib.load('preprocessor_nn.pkl')
    label_encoder = joblib.load('label_encoder_nn.pkl')
    print("Модель и компоненты успешно загружены.")
except FileNotFoundError:
    print("Ошибка: Файлы модели не найдены. Убедитесь, что train_gpu_model.py был запущен и успешно завершился.")
    exit()

# 2. Определение признаков
# Это тот же список признаков, который использовался при обучении
all_features = [
    'апатия', 'кашель', 'лихорадка', 'отказ_от_еды', 'рвота',
    'сыпь', 'хромота', 'диарея', 'насморк', 'чихание', 'боль', 'зуд',
    'полиурия', 'полидипсия', 'вялость', 'тремор', 'судороги', 'анемия',
    'желтуха', 'кровь_в_моче', 'затрудненное_дыхание', 'потеря_веса',
    'отеки', 'увеличение_лимфоузлов', 'гематомы', 'кровотечения',
    'атаксия', 'сухость_глаз', 'облысение', 'одышка', 'кахексия',
    'вид_животного'
]
symptom_features = all_features[:-1]
categorical_features = ['вид_животного']

# 3. Интерактивный ввод данных
print("\n=== Интерфейс для предсказания диагноза ===")

# Ввод вида животного
while True:
    animal_species = input("Введите вид животного (собака/кошка): ").strip().lower()
    if animal_species in ['собака', 'кошка']:
        break
    else:
        print("Неверный ввод. Пожалуйста, выберите 'собака' или 'кошка'.")

# Ввод симптомов
print("\nПожалуйста, введите номера симптомов, которые вы наблюдаете (например, 1 5 12):")
for i, symptom in enumerate(symptom_features):
    print(f"{i+1}. {symptom.replace('_', ' ').capitalize()}")

selected_symptoms_input = input("\nВаш выбор: ")
selected_symptoms_indices = [int(x.strip()) - 1 for x in selected_symptoms_input.split() if x.strip().isdigit() and 0 <= int(x.strip()) - 1 < len(symptom_features)]

# 4. Формирование DataFrame для предсказания
new_data = {symptom: [0] for symptom in symptom_features}
for index in selected_symptoms_indices:
    symptom_name = symptom_features[index]
    new_data[symptom_name] = [1]
    
new_data['вид_животного'] = [animal_species]

input_df = pd.DataFrame(new_data)

# Упорядочиваем колонки
input_df = input_df[all_features]

# 5. Преобразование данных с помощью сохраненного препроцессора
processed_input = preprocessor.transform(input_df)

# 6. Предсказание
predictions = model.predict(processed_input)
top_3_indices = predictions[0].argsort()[-3:][::-1]
top_3_probabilities = predictions[0][top_3_indices]
top_3_diagnoses = label_encoder.inverse_transform(top_3_indices)

print("\n--- Предсказание ---")
print("Наиболее вероятные диагнозы:")
for i in range(len(top_3_diagnoses)):
    diagnosis = top_3_diagnoses[i]
    probability = top_3_probabilities[i]
    print(f"{i+1}. {diagnosis.replace('_', ' ').capitalize()}: {probability*100:.2f}%")