import pandas as pd
import numpy as np
import random
import os
from time import gmtime, strftime

import keras
import tensorflow as tf
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from sklearn.model_selection import ParameterSampler

import data_generator_cnn_videos as dgv

from utils.visualization_utils import plot, visualize_predictions
from utils.hyperparam_utils import create_callbacks, get_optimizer

LOGDIR_TOP = os.path.join('tensorboard_fileversion2\\')
n_runs = 20

def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv.create_datagen()
    # test_best(timestep_num, feature_num, datagen_train, datagen_test)

    logdir = os.path.join(LOGDIR_TOP + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))

    hparams_samples = get_samples(n_runs)
    for sample in hparams_samples: 
        model = create_cnn(timestep_num, feature_num, sample)
        train(model, datagen_train, logdir, sample, datagen_test)


def get_samples(n_runs=10):
    param_grid = {
                  'dropout'         : [i / 100 for i in range(50)],
                  'kernel_size'     : [i for i in range (3, 16)],
                  'kernel_number'   : [i for i in range(3,15)],
                  'max_pool_size'   : [i for i in range(1, 5)],
                  'learning_rate'   : [i / 1e5 for i in range(100)],
                  'optimizer'       : ['Adam', 'RMSProp', 'SGD'], 
                  'layer_number'    : [2, 3]
                 }

    return list(ParameterSampler(param_grid, n_iter=n_runs))


def test_best(timestep_num,
              feature_num,
              datagen_train,
              datagen_test): 
    
    hparams = {
               'kernel_size'           : 3,
               'regularizer'           : 0.1,
               'kernel_number'         : 2
              }

    cnn_model = create_cnn(timestep_num, feature_num, hparams)
    history, model = train(cnn_model, datagen_train, hparams=hparams, datagen_test=datagen_test)
    plot(history)
    visualize_predictions(model, datagen_test)


def train(model, datagen_train, logdir=None, hparams=None, datagen_test=None):
    '''
    '''

    history = model.fit_generator(validation_steps=5,
                                  generator=datagen_train.datagen(),
                                  validation_data=datagen_test.datagen(),
                                  steps_per_epoch=30,
                                  epochs=15,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=create_callbacks(logdir, hparams)
                                  )

    return history, model




def create_cnn(timesteps, features, hparams):
    '''
    '''

    strides= (1,1)

    model = keras.Sequential()
    model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=strides, input_shape=(timesteps, features, 1)))    
    model.add(MaxPooling2D(pool_size=hparams['max_pool_size']))
 
    for _ in range(hparams['layer_number'] - 1):
        try:
            model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=strides))
            model.add(MaxPooling2D(pool_size=hparams['max_pool_size']))

        except ValueError: 
            print('Tried to create a MaxPool Layer that is not possible to create,',
            'because it would lead to negative dimensions. Creation was skipped')

    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':
    main() 

