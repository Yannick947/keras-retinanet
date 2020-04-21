import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from time import gmtime, strftime

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

from tensorboard.plugins.hparams import api as hp

import data_generator_videos as dgv


LOGDIR_TOP = os.path.join('tensorboard\\')


def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv.create_datagen()
    
    hp_domains, metrics = create_hyperparams_domains()
    with tf.summary.create_file_writer(LOGDIR_TOP).as_default():
         hp.hparams_config(
            hparams=list(hp_domains.values()),
            metrics=metrics)

    for num_units in hp_domains['num_units'].domain.values:
        for regularizer in (hp_domains['regularizer'].domain.min_value, hp_domains['regularizer'].domain.max_value):
            for activation in hp_domains['activation'].domain.values:
                logdir = os.path.join(LOGDIR_TOP + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))

                hparams = {
                    'num_units'             : num_units,
                    'regularizer'           : regularizer,
                    'activation'            : activation,
                }

                lstm_model = create_lstm(timestep_num, feature_num, hparams)
                history, model = train(lstm_model, datagen_train, logdir, hparams, datagen_test)

    # plot(history)
    # visualize_predictions(model, datagen_test)


def create_hyperparams_domains(): 
    '''
    '''
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([10, 40, 60, 100]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_ACTIVATION_KERNEL =  hp.HParam('activation', hp.Discrete(['relu', 'sigmoid']))
    HP_KERNEL_REGULARIZER = hp.HParam('num_units', hp.RealInterval(0.0, 0.8))

    HP_TRAIN_LOSS = hp.Metric("loss", group="train", display_name="training loss")
    HP_VAL_LOSS   = hp.Metric("val_loss", group="validation", display_name="validation loss")
                                    
    hp_domains = {'num_units'           : HP_NUM_UNITS,
                  'dropout'             : HP_DROPOUT,
                  'optimizer'           : HP_OPTIMIZER, 
                  'regularizer'         : HP_KERNEL_REGULARIZER, 
                  'activation'          : HP_ACTIVATION_KERNEL,
                  }

    metrics = [HP_TRAIN_LOSS, HP_VAL_LOSS]

    return hp_domains, metrics

def create_callbacks(logdir, hparams=None): 
    '''
    '''
    callbacks = list()
    callbacks.append(tf.keras.callbacks.TensorBoard(logdir))
    callbacks.append(hp.KerasCallback(logdir, hparams))

    return callbacks


def train(model, datagen_train, logdir, hparams=None, datagen_test=None):
    '''
    '''
    history = model.fit_generator(validation_steps=5,
                                generator=datagen_train.datagen(),
                                validation_data=datagen_test.datagen(),
                                steps_per_epoch=30,
                                epochs=10,
                                verbose=1,
                                shuffle=True,
                                callbacks=create_callbacks(logdir, hparams)
                                )

    return history, model


def create_lstm(timesteps, features, hparams):
    '''
    '''
    
    model = tf.keras.Sequential()
    # input_shape is of dimension (timesteps, features)
    model.add(
        LSTM(
            kernel_initializer          = tf.keras.initializers.Zeros(),
            recurrent_initializer       = tf.keras.initializers.Zeros(),
            bias_initializer            = tf.keras.initializers.Zeros(),
            units                       = hparams['num_units'],
            activation                  = hparams['activation'],
            recurrent_activation        = 'sigmoid',
            kernel_regularizer          = tf.keras.regularizers.l2(hparams['regularizer']),
            input_shape                 = (timesteps, features),
            return_sequences            = False))

    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


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

if __name__ == '__main__':
    main() 
