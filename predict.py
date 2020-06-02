import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sys import argv
import cv2
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model

import image_ext as imgext
from numpy_helpers import setup_numeric_floats

PREDICTION_MAP = { 0: 'Apple', 1: 'Orange' }


def get_prediction(model, image_path):
    img = imgext.load_img(img_path) / 255.0

    prediction = model.predict(np.array([
        img,
    ]))

    return prediction[0]


if __name__ == "__main__":
    if len(argv) != 3:
        print('Wrong args count. Must be: `predict.py model.h5 img.png`')
        exit(1)

    setup_numeric_floats()

    model_file_name = argv[1]
    model = load_model(model_file_name)

    img_path = argv[2]

    prediction_output = get_prediction(model, img_path)

    print(prediction_output)

    predicted_label = np.argmax(prediction_output)
    print(PREDICTION_MAP[predicted_label])