import numpy as np
from tensorflow.python.framework import ops
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

physical_devices = tf.config.list_physical_devices('GPU') # Obtener la lista de GPU's instaladas en la maquina
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())


hp_parser = {
    "learning_rate": {0:.0, 1:10e-1, 2:10e-2, 3:10e-3, 4:10e-4, 5:10e-5},
    "dropout": {0:0, 1:.1, 2:.15, 3:.2, 4:.25, 5:.3, 6:.35, 7:.4, 8:.45},
    "units": {0:25, 1:50, 2:75, 3:100, 4:125, 5:150, 6:175, 7:200, 8:225, 9:250, 10:275, 11:300}
}

def adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval):
    """
    Add an extra dimention to use in LSTM networks
    """
    xtrain = np.reshape(xtrain.values, (xtrain.shape[0], xtrain.shape[1], 1))
    ytrain = np.reshape(ytrain.values, (ytrain.shape[0], ytrain.shape[1], 1))
    xval = np.reshape(xval.values, (xval.shape[0], xval.shape[1], 1))
    yval = np.reshape(yval.values, (yval.shape[0], yval.shape[1], 1))
    xtest = np.reshape(xtest.values, (xtest.shape[0], xtest.shape[1], 1))
    ytest = np.reshape(ytest.values, (ytest.shape[0], ytest.shape[1], 1))
    return xtrain, xtest, ytrain, ytest, xval, yval


def fit_lstm_model(xtrain, ytrain, xval, yval, individual_fixed_part, individual_variable_part, prediction_horizon, scaler, epochs=10, batch=1024):
    """
    Train a LSTM model.
    :param individual_fixed_part: Vector of 3 positions [learning_rate {0-3}, dropout {0-5}, #layers (including the first one)].
    :param individual_variable_part: Vector of #layers positions. Each position can be {0-9}
    :return: {loss, mean_absolute_percentage_error, mean_squared_error} in %
    """
    dp = hp_parser["dropout"][individual_fixed_part[1]] / individual_fixed_part[2]
    model = Sequential()
    model.add(LSTM(units=hp_parser["units"][individual_variable_part[0]], return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(Dropout(dp))
    for i in range(1, individual_fixed_part[2]):
        if i < individual_fixed_part[2]-1:
            model.add(LSTM(units=hp_parser["units"][individual_variable_part[i]], return_sequences=True))
        else:
            model.add(LSTM(units=hp_parser["units"][individual_variable_part[i]], return_sequences=False))
        model.add(Dropout(dp))
    model.add(Dense(units=prediction_horizon, activation="tanh"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_parser["learning_rate"][individual_fixed_part[0]]), loss="mean_squared_error", metrics=[keras.metrics.MAPE, keras.metrics.MSE])
    model.fit(x=xtrain, y=ytrain, epochs=epochs, batch_size=batch, verbose=0, validation_data=(xval, yval))
    mse, mape = getMetrics_denormalized(model, xval, yval, batch, scaler)
    return mse, mape, model 


def getMetrics_denormalized(model, xval, yval, batch, scaler):
    predictions = model.predict(xval, batch_size=batch, verbose=0)
    pred = scaler.inverse_transform(predictions.reshape(1, -1)).flatten()
    real = scaler.inverse_transform(yval.reshape(1, -1)).flatten()
    return keras.metrics.MSE(real, pred), keras.metrics.mape(real, pred)

def resetTF():
    ops.reset_default_graph()
    keras.backend.clear_session()

