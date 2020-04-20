from keras.layers import LSTM, Dense
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras as keras
import numpy as np
from sklearn.preprocessing import StandardScaler

import data_generator_videos as dgv


def main():

    timestep_num, feature_num = dgv.get_cleaned_lengths()
    lstm_model = create_lstm(timestep_num, feature_num)
    datagen_train, datagen_test = dgv.create_datagen()
    history = train(lstm_model, datagen_train, datagen_test)
    plot(history)

def train(model, datagen_train, datagen_test=None):

    history = model.fit_generator(validation_steps=1,
                                  generator=datagen_train,
                                  validation_data=datagen_test,
                                  steps_per_epoch=1,
                                  epochs=150,
                                  verbose=1,
                                  shuffle=False)
    return history


# def scale(data, standardize=False):
#     global scaler
    #TODO: Everything

    # if not standardize:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    # else:
    #     scaler = StandardScaler()

    # data = pd.DataFrame(scaler.fit_transform(df_scale))

    # return df_scale.join(data_rest)


def rescale(data):
    global scaler
    return pd.DataFrame(scaler.inverse_transform(data), columns=data.columns)


def create_lstm(timesteps, features):
    model = keras.Sequential()
    # input_shape is of dimension (timesteps, features)
    model.add(
        LSTM(
            units=100,
            activation='relu',
            # recurrent_activation='relu'
            #TODO: check value 0.1 if its too small or too high
            kernel_regularizer = keras.regularizers.l2(0.1),
            input_shape=(
                timesteps,
                features),
            return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


def plot(history):
    # plot history
    plt.plot(history.history['loss'], label='train')
    print(history.history.keys())
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main() 
