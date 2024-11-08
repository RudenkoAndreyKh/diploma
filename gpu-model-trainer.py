from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from pathlib import Path
import numpy as np
import tensorflow as tf

# Переконаємося, що TensorFlow використовує GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU не знайдено. Використовується CPU.")

root_dir = Path(__file__).resolve().parent

plant_resolution = 512

# Оновлений словник з назвами рослин
plantsMap = {
    0: 'African Violet (Saintpaulia ionantha)',
    1: 'Aloe Vera',
    2: 'Anthurium (Anthurium andraeanum)',
    3: 'Areca Palm (Dypsis lutescens)',
    4: 'Asparagus Fern (Asparagus setaceus)',
    5: 'Begonia (Begonia spp.)',
    6: 'Bird of Paradise (Strelitzia reginae)',
    7: 'Birds Nest Fern (Asplenium nidus)',
    8: 'Boston Fern (Nephrolepis exaltata)',
    9: 'Calathea',
    10: 'Cast Iron Plant (Aspidistra elatior)',
    11: 'Chinese evergreen (Aglaonema)',
    12: 'Chinese Money Plant (Pilea peperomioides)',
    13: 'Christmas Cactus (Schlumbergera bridgesii)',
    14: 'Chrysanthemum',
    15: 'Ctenanthe',
    16: 'Daffodils (Narcissus spp.)',
    17: 'Dracaena',
    18: 'Dumb Cane (Dieffenbachia spp.)',
    19: 'Elephant Ear (Alocasia spp.)',
    20: 'English Ivy (Hedera helix)',
    21: 'Hyacinth (Hyacinthus orientalis)',
    22: 'Iron Cross begonia (Begonia masoniana)',
    23: 'Jade plant (Crassula ovata)',
    24: 'Kalanchoe',
    25: 'Lilium (Hemerocallis)',
    26: 'Lily of the valley (Convallaria majalis)',
    27: 'Money Tree (Pachira aquatica)',
    28: 'Monstera Deliciosa (Monstera deliciosa)',
    29: 'Orchid',
    30: 'Parlor Palm (Chamaedorea elegans)',
    31: 'Peace lily',
    32: 'Poinsettia (Euphorbia pulcherrima)',
    33: 'Polka Dot Plant (Hypoestes phyllostachya)',
    34: 'Ponytail Palm (Beaucarnea recurvata)',
    35: 'Pothos (Ivy arum)',
    36: 'Prayer Plant (Maranta leuconeura)',
    37: 'Rattlesnake Plant (Calathea lancifolia)',
    38: 'Rubber Plant (Ficus elastica)',
    39: 'Sago Palm (Cycas revoluta)',
    40: 'Schefflera',
    41: 'Snake plant (Sanseviera)',
    42: 'Tradescantia',
    43: 'Tulip',
    44: 'Venus Flytrap',
    45: 'Yucca',
    46: 'ZZ Plant (Zamioculcas zamiifolia)'
}

# Функція для отримання назви рослини за індексом
def get_plant_name(plant_index):
    return plantsMap.get(plant_index, "Unknown plant")

import warnings
warnings.filterwarnings('ignore')

plant_recognition_dataset_path = root_dir / "house_plant_species"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    plant_recognition_dataset_path,
    subset='training',
    batch_size=128,
    class_mode='categorical',
    target_size=(plant_resolution, plant_resolution)
)

validation_data = datagen.flow_from_directory(
    plant_recognition_dataset_path,
    subset='validation',
    batch_size=128,
    class_mode='categorical',
    target_size=(plant_resolution, plant_resolution)
)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer
predictions = Dense(train_data.num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Metric to monitor
    patience=3,              # Number of epochs to wait for improvement
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,
    verbose=1,
    callbacks=[early_stopping]
)

model.save("plant_recognition_model.keras")

# Функція для передбачення на новому зображенні
def predict_image(image_path):
    # Завантажуємо зображення
    img = image.load_img(root_dir / image_path, target_size=(plant_resolution, plant_resolution))

    # Перетворюємо зображення на масив чисел
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Додаємо ще один вимір для пакета

    # Нормалізація зображення (як і в навчанні)
    img_array /= 255.0

    # Передбачення на новому зображенні
    prediction = model.predict(img_array)
    
    predicted_class = np.argmax(prediction, axis=-1)  # Визначаємо клас з найбільшою ймовірністю
    predicted_probability = np.max(prediction)  # Визначаємо ймовірність

    # Виведення результату
    if predicted_probability > 0.5:
        plant_name = get_plant_name(predicted_class[0])
        print(f"Квітка '{plant_name}' впізнана з ймовірністю {predicted_probability:.2f}")
        return True, plant_name
    else:
        print(f"Квітка не впізнана. Ймовірність: {predicted_probability:.2f}")
        return False, None

# Виклик функції для тестового зображення
predict_image("data/test/1.jpg")
