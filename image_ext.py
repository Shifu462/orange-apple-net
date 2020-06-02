import numpy as np
import cv2
import os
import image_config


def get_channel(cv2img, ch_num=0, invert=False):
    channel = cv2img[:, :, ch_num]

    if invert:
        channel = channel * (-1) + 255

    return channel


def load_img(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, image_config.image_size)

    img = np.array(img).reshape(image_config.input_shape)

    return img.astype('float32')


def load_images_batch(dir):
    ''' у dir должен быть слеш '/' на конце. '''

    return [load_img(dir + i) for i in os.listdir(dir)]
