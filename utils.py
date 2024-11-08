import os
import numpy as np
from tensorflow.keras.preprocessing import image
from variables import (
    root_dir,
    plant_resolution,
    plantsMap,
    plant_recognition_model_path,
    disease_recognition_model_path,
    disease_resolution,
    diseaseOrHealthy,
)
from tensorflow.keras.models import load_model


# Функція для отримання назви рослини за індексом
def get_plant_name(plant_index):
    return plantsMap.get(plant_index, "Unknown plant")


def get_plant_disease(plant_index):
    return diseaseOrHealthy.get(plant_index, "Healthy")


# Функція для передбачення на новому зображенні
def predict_image(image_path):
    # Завантажена модель
    model = None
    print(plant_recognition_model_path)

    if plant_recognition_model_path.exists():
        model = load_model(plant_recognition_model_path)

        # Завантажуємо зображення
        img = image.load_img(
            root_dir / image_path, target_size=(plant_resolution, plant_resolution)
        )

        # Перетворюємо зображення на масив чисел
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Додаємо ще один вимір для пакета

        # Нормалізація зображення (як і в навчанні)
        img_array /= 255.0

        # Передбачення на новому зображенні
        prediction = model.predict(img_array)

        predicted_class = np.argmax(
            prediction, axis=-1
        )  # Визначаємо клас з найбільшою ймовірністю
        predicted_probability = np.max(prediction)  # Визначаємо ймовірність

        # Виведення результату
        if predicted_probability > 0.4:
            plant_name = get_plant_name(predicted_class[0])
            print(
                f"Квітка '{plant_name}' впізнана з ймовірністю {predicted_probability:.2f}"
            )
            predict_disease(image_path, plant_name)
            return True, plant_name, image_path
        else:
            plant_name = get_plant_name(predicted_class[0])
            print(
                f"Квітка {plant_name} не впізнана. Ймовірність: {predicted_probability:.2f}"
            )
            return False, None
    else:
        return False, None


def predict_all():
    # Шлях до директорії
    directoryPath = root_dir / "data" / "test"

    # Отримуємо всі назви файлів у зазначеній директорії
    filesNames = [
        file
        for file in os.listdir(directoryPath)
        if os.path.isfile(os.path.join(directoryPath, file))
    ]

    recognizedPlants = []

    for fileName in filesNames:
        recognizedPlants.append(predict_image(directoryPath / fileName))

    print(recognizedPlants)


def predict_disease(image_path, plant_name="/xxx/"):
    # Завантажена модель
    model = None

    if disease_recognition_model_path.exists():
        model = load_model(disease_recognition_model_path)

        # Завантажуємо зображення
        img = image.load_img(
            root_dir / image_path, target_size=(disease_resolution, disease_resolution)
        )

        # Перетворюємо зображення на масив чисел
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array, axis=0
        )  # Додаємо ще один вимір для пакета

        # Нормалізація зображення (як і в навчанні)
        img_array /= 255.0

        # Передбачення на новому зображенні
        prediction = model.predict(img_array)
        print(prediction)

        predicted_class = np.argmax(
            prediction, axis=-1
        )  # Визначаємо клас з найбільшою ймовірністю
        predicted_probability = np.max(prediction)  # Визначаємо ймовірність

        # Виведення результату
        if predicted_probability > 0.4:
            disease_name = get_plant_disease(predicted_class[0])
            print(
                f"Хвороба квітки {plant_name} '{disease_name}' впізнана з ймовірністю {predicted_probability:.2f}"
            )
            return True, disease_name, image_path
        else:
            disease_name = get_plant_disease(predicted_class[0])
            print(
                f"Хвороба квітки {plant_name} {disease_name} не впізнана. Ймовірність: {predicted_probability:.2f}"
            )
            return False, None
    else:
        return False, None


def predict_all_disease():
    return
