from keras.layers import LSTM, Dense
import pandas as pd
import random
import matplotlib.pyplot as plt
import keras as keras
import numpy as np

import data_generator_videos as dgv

def main():

    timestep_num, feature_num = dgv.get_cleaned_lengths()
    lstm_model = create_lstm(timestep_num, feature_num)
    datagen_train, datagen_test = dgv.create_datagen()
    history, model = train(lstm_model, datagen_train, datagen_test)
    plot(history)
    visualize_predictions(model, datagen_test)


def train(model, datagen_train, datagen_test=None):

    history = model.fit_generator(validation_steps=8,
                                  generator=datagen_train.datagen(),
                                  validation_data=datagen_test.datagen(),
                                  steps_per_epoch=30,
                                  epochs=30,
                                  verbose=1,
                                  shuffle=False)
    return history, model


def create_lstm(timesteps, features):
    model = keras.Sequential()
    # input_shape is of dimension (timesteps, features)
    model.add(
        LSTM(
            kernel_initializer=keras.initializers.Zeros(),
            recurrent_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            units=30,
            activation='tanh',
            recurrent_activation='sigmoid',
            #TODO: check value 0.1 if its too small or too high
            kernel_regularizer = keras.regularizers.l2(0.1),
            input_shape=(timesteps, features),
            return_sequences=False))

    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.adam(), )
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
