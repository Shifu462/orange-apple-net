import os
import cv2
from sys import argv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as keras_backend

import image_config

DEFAULT_MODEL_NAME = 'model'
PATH_SAVED_MODELS = './saved_models'


def createModel(input_shape, output_count=2):
    model = Sequential([
        Conv2D(16, (5, 5), activation='relu', input_shape=input_shape, padding='same', data_format='channels_last'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(output_count, activation='softmax')
    ])

    # model.summary()

    opt = SGD(lr=0.001)  # was 'adam'

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['acc'])

    return model

def save_model(model, model_name):
    if not os.path.exists(PATH_SAVED_MODELS):
        os.mkdir(PATH_SAVED_MODELS)

    model_json = model.to_json()

    with open(PATH_SAVED_MODELS + f'/{model_name}.json', 'w') as json_file:
        json_file.write(model_json)

    model.save(PATH_SAVED_MODELS + f'/{model_name}.h5')

if __name__ == '__main__':

    batch_size = 32
    training_set = ImageDataGenerator(
                        rescale=1./255,
                    ).flow_from_directory('./Dataset/Train',
                        color_mode='rgb',
                        batch_size=batch_size,
                        target_size=image_config.image_size)

    first_batch_labeled = training_set[0]
    first_batch = first_batch_labeled[0]
    first_img = first_batch[0]
    first_img = np.array(first_img)[:, :, ::-1]

    # first_img = cv2.cvtColor(first_batch[0], cv2.COLOR_BGR2RGB)
    # cv2.imshow('random_img', first_img)
    # cv2.waitKey(0)

    print('class_indices', training_set.class_indices)
    print('samples: ', training_set.n)

    model = createModel(
        image_config.input_shape,
        len(training_set.class_indices),
    )

    test_set = ImageDataGenerator(rescale=1./255)\
                    .flow_from_directory('./Dataset/Test',
                        batch_size=batch_size,
                        target_size=image_config.image_size)

    print('--- test samples: ', test_set.n)

    model.fit(training_set,
              steps_per_epoch=training_set.n / batch_size,
              validation_data=test_set,
              validation_steps=3,
              epochs=30,
              verbose=1)

    # model.evaluate(test_set, batch_size=batch_size, verbose=2)

    model_name = argv[1] if len(argv) > 1 else DEFAULT_MODEL_NAME

    save_model(model, model_name)

    print(f'Ok! Saved as "{model_name}".')
