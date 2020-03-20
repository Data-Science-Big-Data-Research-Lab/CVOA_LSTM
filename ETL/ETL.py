import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path_to_data, useNormalization=False):
    """
    Load dataset and normalize
    :param path_to_data: Path to de input dataset
    :return: Normalized dataset as one-column vector and scaler object
    """
    data = pd.read_csv(path_to_data, header=None, engine="python", index_col=0, squeeze=True, usecols=[0,1], parse_dates=[0], date_parser=lambda x: pd.datetime.strptime(x,'%Y-%m-%d %H:%M'))
    data = data.values.astype("float32").reshape(-1, 1)
    scaler = None
    if useNormalization:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
    return data, scaler


def data_to_supervised(data, historical_window, prediction_horizon):
    """
    Convert one-vector data to a lagged matrix.
    :param data: One-vector input
    :param historical_window: Number of past samples to train in the model
    :param prediction_horizon: Number of future values to be forecasted
    :return: A matrix of data
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # Generate past values (t-W, ... t-1)
    for i in range(historical_window, 0, -1):
        cols.append(dff.shift(i))
        names += [('W%d' % (i)) for j in range(n_vars)]
        # Generate future values (t, t+1, ... t+H)
    for i in range(0, prediction_horizon):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('H1') for j in range(n_vars)]
        else:
            names += [('H%d' % (i + 1)) for j in range(n_vars)]
    # Union
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

def splitData(data, historical_window, test_size=.3, val_size=.3):
    """
    Split data into training, validation and test. Also splitted into X and Y
    :param data: Data to be splitted
    :param historical_window: Number of past samples
    :param test_size: Percentaje of the test_size
    :param val_size: Percentaje of the val_size
    :return:
    """
    X = data.iloc[:, 0:historical_window]
    Y = data.iloc[:, historical_window:]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=test_size, random_state=0)
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=val_size, random_state=0)
    return xTrain, xTest, yTrain, yTest, xVal, yVal