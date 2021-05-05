from matplotlib import cm
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from image import Image
from constants import classnames, folder

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.config.threading.set_inter_op_parallelism_threads(6)
# tf.config.threading.set_intra_op_parallelism_threads(6)
# tf.config.set_soft_device_placement(True)


class Covnet():
    """
    Class for a CNN implementation with CIFAR10
    """

    def __init__(self):
        self.image = Image()
        self.buildModel()

    def buildModel(self):
        """
        Generate a model with different layers
        """
        # Basic tensorflow layers for grayscale images
        self.model = keras.Sequential()
        # CNN Layers
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                           input_shape=self.image.getSizeImage()))
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(keras.layers.MaxPool2D((2, 2)))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(keras.layers.MaxPool2D((2, 2)))
        self.model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(keras.layers.MaxPool2D((2, 2)))
        # Dense Layers
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(
            len(classnames), activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def trainModel(self):
        """
        Fit & evaluate the model
        """
        trainImages, trainLabels = self.image.getTrainSet()
        testImages, testLabels = self.image.getTestSet()
        history = self.model.fit(
            trainImages, trainLabels, epochs=10, validation_data=(testImages, testLabels))
        self.summarizeDiagnostic(history)

    def summarizeDiagnostic(self, history):
        """
        Plot loss & accuracy of the model
        """
        # Loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # Accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        # plt.show()
        plt.savefig(folder + 'diagnostic.png')

    def predictTest(self):
        """
        Make predictions with a test set
        """
        images, labels = self.image.getTestSet()
        _, testAcc = self.model.evaluate(images, labels, verbose=2)
        print('\nTest accuracy:', testAcc)

        self.predictions = self.model.predict(images)
        self.plotResults(images, labels)

    def plotImage(self, i, predictions, trueLabel, img):
        """
        docstring
        """
        trueLabel, img = trueLabel[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

        predictedLabel = np.argmax(predictions)
        if predictedLabel == trueLabel:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel('{} {:2.0f}% ({})'.format(
            classnames[predictedLabel], 100 * np.max(predictions), classnames[trueLabel], color=color))

    def plotValueArray(self, i, predictions, trueLabel):
        """
        docstring
        """
        trueLabel = trueLabel[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions, color='#777777')
        plt.ylim([0, 1])
        predictedLabel = np.argmax(predictions)

        thisplot[predictedLabel].set_color('red')
        thisplot[trueLabel].set_color('blue')

    def plotResults(self, images, labels):
        """
        docstring
        """
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plotImage(i, self.predictions[i], labels, images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plotValueArray(i, self.predictions[i], labels)
        plt.tight_layout()
        # plt.show()
        plt.savefig(folder + 'results.png')


if __name__ == "__main__":
    cnn = Covnet()
    cnn.buildModel()
    cnn.trainModel()
    cnn.predictTest()
