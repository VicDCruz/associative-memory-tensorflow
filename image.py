from matplotlib import cm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

classnames = ['airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Image():
    """
    Class for managing a image library from tensorflow.
    Specifically, we'll use CIFAR10
    """
    
    def __init__(self, gray=False):
        """
        Start getting the image by instantiate an Image object
        """
        self.getImages()
        if (gray):
            self.rgb2grayscale()
        self.showSamples()

    def plotSample(self, index, gray=True):
        """
        Display one image from dataset by index
        """
        if (gray):
            plt.imshow(self.trainImages[index], cmap=plt.cm.binary)
        else:
            plt.imshow(self.trainImages[index])
        plt.colorbar()
        plt.grid(False)

    def getImages(self):
        """
        Retrieve dataset of images & display a summary
        """
        print('Get images...')
        cifar = keras.datasets.cifar10
        (self.trainImages, trainLabels), (self.testImages, testLabels) = cifar.load_data()
        # Labels form: [[1], [2], [3], ...]
        self.trainLabels = trainLabels.reshape(-1, ) # From 2D to 1D array
        self.testLabels = testLabels.reshape(-1, ) # From 2D to 1D array
        # Labels form: [1, 2, 3, ...]

        self.sizeImage = self.trainImages[0].shape

        print('=== Summary ===')
        print('Dimensions of train image set: {0}'.format(self.trainImages.shape))
        print('Dimensions of test image set: {0}'.format(self.testImages.shape))
        print('Total train labels: {0}'.format(len(self.trainLabels)))
        print('Total test labels: {0}'.format(len(self.testLabels)))
        print('Image size: {0}'.format(self.sizeImage))
        print('=== Summary ===')

    def rgb2grayscale(self):
        """
        Transform both image sets from RGB to grayscale
        """
        example = int(np.random.random() * 2)

        print('=== Transform to grayscale===')
        plt.figure()
        plt.subplot(1, 2, 1)
        self.plotSample(example, gray=False)

        self.trainImages = np.mean(self.trainImages, axis=3) / 255
        self.testImages = np.mean(self.testImages, axis=3) / 255

        print('Dimensions of train image set: {0}'.format(self.trainImages.shape))
        print('Dimensions of test image set: {0}'.format(self.testImages.shape))

        print('=== Transform to grayscale===')
        plt.subplot(1, 2, 2)
        self.plotSample(example)
        plt.show()

    def showSamples(self):
        """
        Display 25 subplots of the train image set
        """
        print('Showing sample of images...')
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            self.plotSample(i)
            plt.title(classnames[self.trainLabels[i]])
        plt.show()

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=size))
# print(model.output_shape)

if __name__ == "__main__":
    image = Image()
