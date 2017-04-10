# Jiaming Zhao
# test catsdogs model on single image

from keras.models import load_model
import cv2, numpy as np


im = cv2.resize(cv2.imread('catdog.jpg'), (150, 150)).astype(np.float32)
im = im.transpose((0, 1, 2))
im = np.expand_dims(im, axis=0)

model = load_model('cats_dogs_model.h5')
out = ( model.predict(im) )[0][0]

cat_dog = 'dog' if round( out ) == 1 else 'cat'
print(out)
print(cat_dog)
