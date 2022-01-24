import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,\
   Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#function preparing data
def prepare_data(data):
    (train_data, _), (test_data, _) = data

    train_data = normalize_pixels(train_data)
    test_data = normalize_pixels(test_data)

    return train_data, test_data


def normalize_pixels(x):
    return x.astype('float32') / 255.0

#adding noise to image
def add_noise(data):
    noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
    data = data + noise
    # data 0 or 1
    data = np.clip(data, 0., 1.)
    return data

#function that returing model of autoencoder
def autoencoder():
    input = Input(shape=(32, 32, 3), name='dae_input')
    conv_block1 = conv_block(input, 32, 3)
    conv_block2 = conv_block(conv_block1, 64, 3)
    conv_block3 = conv_block(conv_block2, 128, 3)
    conv_block4 = conv_block(conv_block3, 128, 3,1)



    deconv_block1 = deconv_block(conv_block4, 128, 3)
    concat1 = Concatenate()([deconv_block1, conv_block2])
    deconv_block2 = deconv_block(concat1, 64, 3)
    concat2 = Concatenate()([deconv_block2, conv_block1])
    deconv_block3 = deconv_block(concat2, 32, 3)


    deconv = Conv2DTranspose(filters=3,
                               kernel_size=3,
                               padding='same')(deconv_block3)

    output = Activation('sigmoid', name='dae_output')(deconv)
    return Model(input, output, name='dae')

#function that return one conv_block
def conv_block(x, filters, kernel_size, strides=2):
   x = Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x
#function that return one deconv_block
def deconv_block(x, filters, kernel_size):
   x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x


def print_images(test_data,test_data_noisy,test_data_denoised):
    idx = 10
    plt.subplot(1, 3, 1)
    plt.imshow(test_data[idx])
    plt.title('normal')
    plt.subplot(1, 3, 2)
    plt.imshow(test_data_noisy[idx])
    plt.title('with noise')
    plt.subplot(1, 3, 3)
    plt.imshow(test_data_denoised[idx])
    plt.title('after denoising')
    plt.show()
    plt.savefig('miw8.png')

def main():
    with tf.device('/GPU:0'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        train_data, test_data = prepare_data(datasets.cifar10.load_data())

        print(train_data)

        train_data_noisy = add_noise(train_data)
        test_data_noisy = add_noise(test_data)

        # i = 10
        # plt.subplot(1, 2, 1)
        # plt.imshow(train_data[i])
        # plt.title('Image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(train_data_noisy[i])
        # plt.title('Image with noise')
        # plt.show()

        model = autoencoder()

        model.compile(loss='mse', optimizer='adam')

        checkpoint = ModelCheckpoint('../model_miw8.h5', verbose=1, save_best_only=True, save_weights_only=True)

        model.fit(train_data_noisy,
                train_data,
                validation_data=(test_data_noisy, test_data),
                epochs=20,
                batch_size=128,
                callbacks=[checkpoint])
        model.save('model_miw8.h5')
        #
        model.load_weights('model_miw8.h5')
        test_data_denoised = model.predict(test_data_noisy)


        print_images(test_data,test_data_noisy,test_data_denoised)

if __name__ == '__main__':
    main()
