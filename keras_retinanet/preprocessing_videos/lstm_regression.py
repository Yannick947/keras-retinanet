import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

from tensorboard.plugins.hparams import api as hp

import data_generator_videos as dgv


LOGDIR = os.path.join('tensorboard')


def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv.create_datagen()
    
    hp_domains = create_hyperparams_domains()

    for num_units in hp_domains['num_units'].domain.values:
        for dropout_rate in (hp_domains['dropout'].domain.min_value, hp_domains['dropout'].domain.max_value):
            for optimizer in hp_domains['optimizer'].domain.values:
                hparams = {
                    'num_units': num_units,
                    'dropout': dropout_rate,
                    'optimizer': optimizer,
                }
                lstm_model = create_lstm(timestep_num, feature_num, hparams)
                history, model = train(lstm_model, datagen_train, hparams, datagen_test)

    
    plot(history)
    visualize_predictions(model, datagen_test)


def create_hyperparams_domains(): 
    '''
    '''
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    hp_domains = {'num_units':HP_NUM_UNITS, 'dropout':HP_DROPOUT, 'optimizer':HP_OPTIMIZER}
    METRIC_VAL_LOSS = 'val_loss'

    #TODO: Check if this is necessary when using callbacks
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_VAL_LOSS, display_name='val_loss')],
        )   

    return hp_domains

def create_callbacks(logdir=LOGDIR, hparams=None): 
    '''
    '''
    callbacks = list()
    callbacks.append(tf.keras.callbacks.TensorBoard(logdir))
    callbacks.append(hp.KerasCallback(logdir, hparams))

    return callbacks



def train(model, datagen_train, hparams=None, datagen_test=None):
    '''
    '''
    history = model.fit_generator(validation_steps=8,
                                generator=datagen_train.datagen(),
                                validation_data=datagen_test.datagen(),
                                steps_per_epoch=30,
                                epochs=10,
                                verbose=1,
                                shuffle=True, 
                                callbacks=create_callbacks(LOGDIR, hparams))
    return history, model


def create_lstm(timesteps, features, hparams):
    '''
    '''
    
    model = tf.keras.Sequential()
    # input_shape is of dimension (timesteps, features)
    model.add(
        LSTM(
            kernel_initializer=tf.keras.initializers.Zeros(),
            recurrent_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            units=hparams['num_units'],
            activation='tanh',
            recurrent_activation='sigmoid',
            #TODO: check value 0.1 if its too small or too high
            # kernel_regularizer = keras.regularizers.l2(0.3),
            input_shape=(timesteps, features),
            return_sequences=False))

    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer=hparams['optimizer'])
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
