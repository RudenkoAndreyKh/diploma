from utils import predict_image, predict_all, predict_disease, predict_all_disease
import argparse
from plant_recognition.recognition_train import recognition_train
from leaves_disease_recognition.disease_train import disease_recognition_train

# Налаштування argparse для прийому параметрів
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Вибір функції для виконання")
    
    # Додаємо параметр командного рядка --mode
    parser.add_argument('--mode', type=str, required=True, choices=['train-plants', 'predict-plant', 'predict-all-plants', 'train-disease', 'predict-plant-disease', 'predict-all-plants-disease'], help='Режим роботи: "train" або "predict"')
    
    # Парсимо аргументи
    args = parser.parse_args()

    # Перевірка значення параметра і виклик відповідної функції
    if args.mode == 'train-plants':
        recognition_train()
    elif args.mode == 'train-disease':
        disease_recognition_train()
    elif args.mode == 'predict-plant':
        predict_image("data/test/aloe-problems.jpg")
    elif args.mode == 'predict-all-plants':
        predict_all()
    elif args.mode == 'predict-plant-disease':
        predict_disease("data/test/african-violet-Diseases.jpeg")
    elif args.mode == 'predict-all-plants-disease':
        predict_all_disease()