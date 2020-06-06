import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from sys import argv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as keras_backend, callbacks as keras_callbacks

import image_config

DEFAULT_MODEL_NAME = 'model'
PATH_SAVED_MODELS = './saved_models'
LOG_DIR = './log/'

target_size = (64, 64)
input_shape = (64, 64, len('rgb'))

def createModel(input_shape):
    layers = [
        Conv2D(16, 3, activation='relu', input_shape=input_shape),

        MaxPooling2D(2),

        Conv2D(32, 3, activation='relu'),

        MaxPooling2D(2),

        Flatten(),

        Dense(256, activation='relu'),

        Dense(1, activation='sigmoid'),
    ]

    model = Sequential(layers)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['acc'])

    return model

def save_model(model, model_name):
    if not os.path.exists(PATH_SAVED_MODELS):
        os.mkdir(PATH_SAVED_MODELS)

    model_json = model.to_json()

    with open(PATH_SAVED_MODELS + f'/{model_name}.json', 'w') as json_file:
        json_file.write(model_json)

    model.save(PATH_SAVED_MODELS + f'/{model_name}.h5')


def createTrainTest(target_size):
    batch_size = 32

    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('./Dataset/Train',
                        batch_size=batch_size,
                        class_mode='binary',
                        target_size=target_size)

    test_set = test_datagen.flow_from_directory('./Dataset/Test',
                        batch_size=batch_size,
                        class_mode='binary',
                        target_size=target_size)

    return (training_set, test_set)


if __name__ == '__main__':
    print('\n')

    train_set, test_set = createTrainTest(target_size)

    model = createModel(input_shape)
    model.fit(train_set,
            steps_per_epoch=16,
            epochs=10,
            validation_data=test_set,
            validation_steps=50,
        )

    model_name = argv[1] if len(argv) > 1 else DEFAULT_MODEL_NAME

    save_model(model, model_name)

    print(f'Ok! Saved as "{model_name}".')
