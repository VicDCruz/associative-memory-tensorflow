from matplotlib import cm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from image import Image
from constants import classnames

image = Image(gray=True)

# Basic tensorflow layers for grayscale images
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=image.getSizeImage()))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(len(classnames), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

images, labels = image.getTrainSet()
model.fit(images, labels, epochs=10)

images, labels = image.getTestSet()
testLoss, testAcc = model.evaluate(images, labels, verbose=2)
print('\nTest accuracy:', testAcc)
