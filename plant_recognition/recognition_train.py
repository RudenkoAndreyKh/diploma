from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from variables import plant_resolution, plant_recognition_dataset_path, batchSize
import warnings

# Переконаємося, що TensorFlow використовує GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU не знайдено. Використовується CPU.")

warnings.filterwarnings('ignore')

def recognition_train():
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
        batch_size=batchSize,
        class_mode='categorical',
        target_size=(plant_resolution, plant_resolution)
    )

    validation_data = datagen.flow_from_directory(
        plant_recognition_dataset_path,
        subset='validation',
        batch_size=batchSize,
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
