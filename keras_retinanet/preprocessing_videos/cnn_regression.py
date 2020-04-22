import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import keras 

import tensorflow as tf
from keras.layers import LSTM, Dense, MaxPooling2D, Conv2D, Flatten

from tensorboard.plugins.hparams import api as hp

import data_generator_cnn_videos as dgv


LOGDIR_TOP = os.path.join('tensorboard\\')


def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv.create_datagen()
    # test_best(timestep_num, feature_num, datagen_train, datagen_test)
    hp_domains, metrics = create_hyperparams_domains()    

    for kernel_size in hp_domains['kernel_size'].domain.values:
        for regularizer in (hp_domains['regularizer'].domain.min_value, hp_domains['regularizer'].domain.max_value):
            for kernel_number in hp_domains['kernel_number'].domain.values:
                for max_pool in hp_domains['max_pool_size'].domain.values:
                    logdir = os.path.join(LOGDIR_TOP + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))

                    hparams = {
                            'kernel_size'           : kernel_size,
                            'regularizer'           : regularizer,
                            'kernel_number'         : kernel_number,
                            'max_pool_size'         : max_pool, 
                            }
                
                    cnn_model = create_cnn(timestep_num, feature_num, hparams)
                    train(cnn_model, datagen_train, logdir, hparams, datagen_test)


                
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


def create_hyperparams_domains(): 
    '''
    '''
    HP_REGULARIZER = hp.HParam('regularizer', hp.RealInterval(0.1, 0.3))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_KERNEL_NUMBER =  hp.HParam('kernel_number', hp.Discrete([2, 4, 8]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 5]))
    HP_MAX_POOL_SIZE = hp.HParam('max_pool_size', hp.Discrete([2, 4]))

    HP_TRAIN_LOSS = hp.Metric("loss", group="train", display_name="training loss")
    HP_VAL_LOSS   = hp.Metric("val_loss", group="validation", display_name="validation loss")
                                    
    hp_domains = {'kernel_size'           : HP_KERNEL_SIZE,
                  'dropout'               : HP_DROPOUT,
                  'optimizer'             : HP_OPTIMIZER, 
                  'regularizer'           : HP_REGULARIZER, 
                  'kernel_number'         : HP_KERNEL_NUMBER,
                  'max_pool_size'         : HP_MAX_POOL_SIZE,
                  }

    metrics = [HP_TRAIN_LOSS, HP_VAL_LOSS] 

    return hp_domains, metrics

def create_callbacks(logdir, hparams=None): 
    '''
    '''

    if logdir == None: 
        return None

    callbacks = list()
    tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = logdir,
            update_freq            = 16, 
            profile_batch          = 0
        )
    callbacks.append(tensorboard_callback)
    callbacks.append(hp.KerasCallback(logdir, hparams))

    return callbacks


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


def visualize_predictions(model, datagen_test):
    datagen_test.reset_label_states()

    predictions = model.predict_generator(generator=datagen_test.datagen(), steps=10)
    predictions_inverse = datagen_test.scaler.scaler_labels.inverse_transform(predictions)

    y_true = datagen_test.scaler.scaler_labels.inverse_transform(datagen_test.get_labels())

    plt.scatter(predictions_inverse, y_true)
    plt.title('Predictions over ground truth')
    plt.xlabel('Predictions')
    plt.ylabel('Ground truth')

    plt.xticks(np.arange(min(np.append(predictions_inverse, y_true)),
                         max(np.append(predictions_inverse , y_true))))

    plt.yticks(np.arange(min(np.append(predictions_inverse, y_true)),
                         max(np.append(predictions_inverse , y_true))))
    plt.show()


def plot(history):
    # plot history
    plt.plot(history.history['loss'], label='train')
    print(history.history.keys())
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def create_cnn(timesteps, features, hparams):
    '''
    '''

    strides= (1,1)

    model = keras.Sequential()
    model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=strides, input_shape=(timesteps, features, 1)))    
    model.add(MaxPooling2D(pool_size=hparams['max_pool_size']))
    model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=strides))
    model.add(MaxPooling2D(pool_size=hparams['max_pool_size']))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

if __name__ == '__main__':
    main() 

