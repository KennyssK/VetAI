# Используем старый, совместимый образ TensorFlow
FROM tensorflow/tensorflow:2.11.0-gpu

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем все необходимые библиотеки в одну команду
# Это помогает избежать конфликтов версий
RUN pip install --no-cache-dir \
    tensorflow==2.11.0 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    numpy==1.24.4 \
    joblib==1.3.2 \
    streamlit \
    Flask \
    gunicorn \
    shap \
    matplotlib \
    protobuf==3.20.0 \
    tqdm \
    fpdf

# Копируем файлы в рабочую директорию
COPY . /app