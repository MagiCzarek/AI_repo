import numpy as np
import argparse
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout


from keras.models import load_model
	"""
	Script of Simple classifier
	
	"""


def pars_arguments():
    parser = argparse.ArgumentParser(description='Script for Convolutional Neural Network ')
    parser.add_argument('-i', '--input', type=str, required=True,
                        nargs='+', help='video path')

    return parser.parse_args()


def prepare_data(data):
    (X_train, Y_train), (X_test, Y_test) = data


    X_train = normalize_pixels(X_train)
    Y_train = to_two_classes(Y_train)
    X_test = normalize_pixels(X_test)
    Y_test = to_two_classes(Y_test)
    print(X_train)
    print(Y_train)

    X_concatenate = np.concatenate((X_train, X_test))
    Y_concatenante = np.concatenate((Y_train, Y_test))

    # X = data[0][0].astype('float32') / 255.0
    # y = to_categorical(data[0][1])

    # for i in range(10):
    #     print(y[i])
    #     plt.imshow(X[i])
    #     plt.show()

    train_images, test_images, train_labels, test_labels = train_test_split(X_concatenate, Y_concatenante,
                                                                            test_size=0.3, random_state=1234)

    return train_images, test_images, train_labels, test_labels


def to_two_classes(y):
    for i in range(len(y)):
        if y[i] == 0 or y[i] == 1 or y[i] == 8 or y[i] == 9:
            y[i] = 1
        else:
            y[i] = 0

    return y


def normalize_pixels(x):
    return x.astype('float32') / 255.0


def main():


    train_images, test_images, train_labels, test_labels = prepare_data(cifar10.load_data())
    class_names = ['animal', 'vehicle']
	
	"""
	Testing on 3 models with different layers
	
	"""

    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history1 = model.fit(train_images, train_labels, epochs=20,
                         validation_data=(test_images, test_labels))
    model.save('model.h5')

    model2 = Sequential()

    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(2, activation='softmax'))

    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history2 = model2.fit(train_images, train_labels, epochs=20,
                          validation_data=(test_images, test_labels))

    model2.save('model2.h5')
    #
    model3 = Sequential()

    model3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model3.add(layers.MaxPooling2D((2, 2)))
    model3.add(Conv2D(64, (3, 3), activation='relu'))
    model3.add(layers.MaxPooling2D((2, 2)))
    model3.add(Conv2D(128, (3, 3), activation='relu'))
    model3.add(Flatten())
    model3.add(Dense(2, activation='softmax'))

    model3.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    history3 = model3.fit(train_images, train_labels, epochs=20,
                          validation_data=(test_images, test_labels))

    model3.save('model3.h5')

    modelfinal = keras.models.load_model('model.h5'.format(1))
    modelfinal2 = keras.models.load_model('model2.h5'.format(1))
    modelfinal3 = keras.models.load_model('model3.h5'.format(1))

    draw_model(history1, test_images, test_labels, modelfinal)
    draw_model(history2, test_images, test_labels, modelfinal2)
    draw_model(history3, test_images, test_labels, modelfinal3)

    print_accuracy(modelfinal, test_images, test_labels)
    print_accuracy(modelfinal2, test_images, test_labels)
    print_accuracy(modelfinal3, test_images, test_labels)


def draw_model(history, test_images, test_labels, model):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def print_accuracy(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
    main()
