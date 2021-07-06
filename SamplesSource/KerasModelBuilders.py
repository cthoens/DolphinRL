from gym import Env

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as K


def conv1_model(env: Env) -> keras.Model:
    """
    Build a keras model that uses a single convolutional layer and can accept an observation of env as a input:
        - Convolution with 32 filters and relu activation
        - Fully connected layer with 128 neurons and relu activation
        - Dense layer with one neuron per action and linear activation
    :param env: The environment to build the model for
    :return: The compiled Keras model
    """
    #if K.image_data_format() == 'channels_first':
    input_shape = (1, ) + env.observation_space.shape
    #else:
    #    input_shape = env.observation_space.shape + (1, )

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(64, 64),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['mae'])
    return model


def conv2_model(env: Env) -> keras.Model:
    """
    Build a keras model that uses two convolutional layers and can accept an observation of env as a input:
        - Convolution with 32 filters and relu activation
        - 2x2 max pooling layer
        - Dropout with rate 0.25
        - Fully connected layer with 128 neurons and relu activation
        - Dropout with rate 0.5
        - Dense layer with one neuron per action and linear activation
    :param env: The environment to build the model for
    :return: The compiled Keras model
    """
    # if K.image_data_format() == 'channels_first':
    #input_shape = (1, ) + env.observation_space.shape
    # else:
    #    input_shape = env.observation_space.shape + (1, )

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(64, 64),
                     activation='relu',
                     input_shape=env.observation_space.shape))
    model.add(Conv2D(32, (32, 32), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(env.action_space.n, activation='linear'))

    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['mae'])
    return model
