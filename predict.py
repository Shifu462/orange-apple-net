import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sys import argv
import tensorflow as tf
from tensorflow.keras.models import load_model

import image_ext as imgext
from numpy_helpers import setup_numeric_floats

PREDICTION_MAP = { 0: 'Apple', 1: 'Orange' }

def get_prediction(model, image_path):
    img = imgext.load_img(img_path, (64, 64))

    prediction = model.predict(np.array([
        img,
    ]))

    return prediction[0][0]


if __name__ == "__main__":
    if len(argv) != 3 and not FORCED_IMG and not FORCED_MODEL_NAME:
        print('Wrong args count. Must be: `predict.py model.h5 img.png`')
        exit(1)

    setup_numeric_floats()

    model_file_name = argv[1]
    img_path = argv[2]

    model = load_model(model_file_name)

    pred = get_prediction(model, img_path)
    pred_label = 'Orange' if pred > 0.5 else 'Apple'

    print(pred_label)
