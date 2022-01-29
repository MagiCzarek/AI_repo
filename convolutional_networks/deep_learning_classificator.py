import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout

'''
Script: Classification for 2 classes animals and vehicles, using model with 3 convolutional layers
Made by Cezary Boguszewski
'''


def prepare_data(data):
    (X_train, Y_train), (X_test, Y_test) = data

    X_train = normalize_pixels(X_train)
    Y_train = to_two_classes(Y_train)
    X_test = normalize_pixels(X_test)
    Y_test = to_two_classes(Y_test)

    X_concatenate = np.concatenate((X_train, X_test))
    Y_concatenante = np.concatenate((Y_train, Y_test))

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

    # model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64), activation='relu')
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    with tf.device('/device:GPU:0'):
        history3 = model.fit(train_images, train_labels, epochs=20,
                             validation_data=(test_images, test_labels))

    model.save('model.h5')

    modelfinal = tf.keras.models.load_model('model.h5'.format(1))

    draw_model(history3, test_images, test_labels, modelfinal)

    print_accuracy(modelfinal, test_images, test_labels)


# ploting accuraccy
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
