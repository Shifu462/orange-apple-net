import cv2

def load_img(filename, size=(64, 64)):
    img = cv2.imread(filename).astype('float32') / 255.0
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
