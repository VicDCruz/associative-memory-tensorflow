from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from constants import classnames, folder


class Image():
    """
    Class for managing a image library from tensorflow.
    Specifically, we'll use CIFAR10
    """

    def __init__(self, gray=False, showSamples=False):
        """
        Start getting the image by instantiate an Image object
        """
        self.retrieveImages()
        if (gray):
            self.rgb2grayscale()
        if (showSamples):
            self.showSamples(gray)

    def plotSample(self, index, gray=True):
        """
        Display one image from dataset by index
        """
        if (gray):
            plt.imshow(self.trainImages[index], cmap=plt.cm.binary)
            plt.colorbar()
        else:
            plt.imshow(self.trainImages[index])
        plt.grid(False)
        plt.title(classnames[self.trainLabels[index]])

    def retrieveImages(self):
        """
        Retrieve dataset of images & display a summary
        """
        print('Get images...')
        cifar = keras.datasets.cifar10
        (self.trainImages, trainLabels), (self.testImages,
                                          testLabels) = cifar.load_data()
        self.trainImages = self.trainImages.astype('float32') / 255
        self.testImages = self.testImages.astype('float32') / 255
        # Labels form: [[1], [2], [3], ...]
        self.trainLabels = trainLabels.reshape(-1, )  # From 2D to 1D array
        self.testLabels = testLabels.reshape(-1, )  # From 2D to 1D array
        # Labels form: [1, 2, 3, ...]

        self.sizeImage = self.trainImages[0].shape

        print('=== Summary ===')
        print('Dimensions of train image set: {0}'.format(
            self.trainImages.shape))
        print('Dimensions of test image set: {0}'.format(
            self.testImages.shape))
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

        self.trainImages = np.mean(self.trainImages, axis=3)
        self.testImages = np.mean(self.testImages, axis=3)

        self.sizeImage = self.trainImages[0].shape

        print('Dimensions of train image set: {0}'.format(
            self.trainImages.shape))
        print('Dimensions of test image set: {0}'.format(
            self.testImages.shape))

        print('Image size: {0}'.format(self.sizeImage))

        print('=== Transform to grayscale===')
        plt.subplot(1, 2, 2)
        self.plotSample(example)
        # plt.show()
        plt.savefig(folder + 'gray.png')

    def showSamples(self, gray):
        """
        Display 25 subplots of the train image set
        """
        print('Showing sample of images...')
        plt.figure(figsize=(10, 9))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            self.plotSample(i, gray)
        # plt.show()
        plt.savefig(folder + 'samples.png')

    def getTrainSet(self):
        """
        Return train image & label set
        """
        return self.trainImages, self.trainLabels

    def getTestSet(self):
        """
        Return test image & label set
        """
        return self.testImages, self.testLabels

    def getSizeImage(self):
        """
        Return size of an image
        """
        return self.sizeImage

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=size))
# print(model.output_shape)


if __name__ == "__main__":
    image = Image(showSamples=True)
