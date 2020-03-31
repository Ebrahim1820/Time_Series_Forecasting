import pickle
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from pandas import concat
from datetime import datetime
import pandas as dropna
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, Activation
from keras.layers import Dropout
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from bayes_opt import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
matplotlib.style.use('ggplot')
sns.set({'figure.figsize': (11, 4)})


N_INPUTS = 20
N_OUTPUTS = 10
# By default VERBOSE, EPOCHS, BATCHSIZE , N_INPUTS = 2, 300, 122-125, 20 is good config for
# predicts N_OUTPUTS = 10 NEURONS= 310
VERBOSE= 2
NEURONS = 300
TRAIN_NUMBERS = 2200 # number of data for training
DATASET = 'Now_Power_Q_Phase_2_value'

PARAMETERS = {'epochs': [100], 'batch_size':[123, 124]}


def pickle_data_load(path):
    # path=('D:/sensor_13050091.pkl','rb')
    Sensor_Data = pickle.load(open(path, 'rb'))
    Sensor_Data = pd.DataFrame.from_dict(Sensor_Data).dropna()
    df = pd.DataFrame.from_dict(Sensor_Data).dropna()
    # df = df.drop(df.columns[1:])
    dates = pd.to_datetime(df['date'])
    df.index = dates
    df = df.sort_index()
    return df


def grap_special_cols(df):
    features = ['Now_Power_Q_Phase_1_value', 'Now_Power_Q_Phase_2_value', 'Now_Power_Q_Phase_3_value']
    df = df[features]
    df.to_csv('Power_Q_Phase.csv')
    return df


def resampledata_from_second_to_minutes(data):
    # first arrange columns 2,1,3 -->1,2,3
    cols = data.columns.tolist()
    data = data[cols[0:]]
    # resample seconds to each minutes
    min_groups = data.resample('60S').sum() # how='ohlc'

    return min_groups

def plot_data(data):
    df = data
    fig, ax = plt.subplots()
    min_groups.plot(ax=ax, linewidth=0.5)
    ax.legend(loc='upper left')
    ###### separate each columns of data in plot
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(len(df.columns)):
        plt.subplot(len(df.columns), 1, i + 1)
        name = df.columns[i]
        plt.plot(df[name])
        plt.title(name, y=0, loc='right')
        plt.yticks([])

    plt.show()
    fig.tight_layout()

# sumerize scores
def summerize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))
    return name, score, s_scores


def split_dataset_(data, N_OUTPUTS, TRAIN_NUMBERS,DATASET):

    print(list(data.columns))
    series = data[DATASET].values[1:]

    #train, test = series[0:2400], series[2400:-18]
    train, test = series[0:TRAIN_NUMBERS], series[TRAIN_NUMBERS:-18]

    if not (len(train) / N_OUTPUTS).is_integer() or not (len(test) / N_OUTPUTS).is_integer():
        raise Exception('The split number must be an integer number')

    train = np.array(np.split(train, len(train) / N_OUTPUTS))
    test = np.array(np.split(test, len(test) / N_OUTPUTS))

    return train, test

    test = np.array(test)
    X_test, y_test = [], []
    for i in range(N_INPUTS, len(test) - N_OUTPUTS):
        X_test.append(test[i - N_INPUTS: i])
        y_test.append(test[i: i + N_OUTPUTS])
    X_test, y_test = np.array(X_test), np.array(y_test)
    return X_train, y_train, X_test, y_test


def series_to_supervised(train, N_INPUTS, N_OUTPUTS):
    train = train.ravel()
    X, y = list(), list()
    in_start = 0
    # input sequence (t-n,...,t-1)
    for _ in range(len(train)):
        in_end = in_start + N_INPUTS
        out_end = in_end + N_OUTPUTS
        # ensure we have enough data for this instance
        if out_end <= len(train):
            x_input = train[in_start:in_end]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(train[in_end: out_end])

        # move along one time step
        in_start += 1

    return np.array(X), np.array(y)


def model_net():

    model = Sequential()
    model.add(LSTM(units=NEURONS, activation='relu', input_shape=(N_INPUTS, 1), name="LSTM_1"))
    #model.add(LSTM(100, activation='relu', name='LSTM_2'))
    # tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True , scale=True)
    model.add(Dense(100, activation='relu', name='Dense_1'))
    # model.add(Dropout(0.2))
    model.add(Dense(N_OUTPUTS, name='Dense_2'))
    # model.add(Activation('linear'))# n minutes
    model.compile(loss='mean_squared_error', optimizer='adam')
    #monitor = EarlyStopping(monitor='loss', min_delta=1e-3,
                            #patience=100, verbose=0, mode='auto', restore_best_weights=True)
    #epochs = monitor.stopped_epoch
    #epochs_needed.append(epochs)
    #tf.keras.backend.clear_session()
    return model

# make a forecast
def forecast(model, history, N_INPUTS):
    # flatten data
    data = np.array(history)
    # flatten data
    data = data.ravel()
    # retrieve last observation for input data
    input_x = data[-N_INPUTS:]
    print(input_x.shape)
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the nexr hour
    yhat = model.predict(input_x)
    # we only want the vector forecast
    yhat = yhat[0]

    return yhat

def evaluate_forecasts(actual_data, forecast_data):
    scores = list()
    # calculate an RMSE score for each minuts
    for i in range(actual_data.shape[1]):
        mse = mean_squared_error(actual_data[:, i], forecast_data[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    print("total scores", scores)
    # calculate overal RMSE
    s = 0
    for row in range(actual_data.shape[0]):
        for col in range(actual_data.shape[1]):
            s += (actual_data[row, col] - forecast_data[row, col]) ** 2

    score = sqrt(s / (actual_data.shape[0] * actual_data.shape[1]))

    return score, scores

def hyp_param_Optimize(train, N_INPUTS,N_OUTPUTS,NEURONS,PARAMETERS):
    train_X, train_y = series_to_supervised(train, N_INPUTS, N_OUTPUTS)
    model = KerasRegressor(build_fn=model_net, verbose=2)
    model = GridSearchCV(estimator=model, param_grid= PARAMETERS, n_jobs=1, cv=3)
    grid_results = model.fit(train_X, train_y)

    print("Best scores & parameters: ", model.best_score_, model.best_params_)
    means = model.cv_results_['mean_test_score']
    parameters1 = model.cv_results_['params']

    for mean, parameter in zip(means, parameters1):
        print('mean: ', mean, " <--> ", ' parameter: ',  parameter)

    return model

def evaluate_model(train, test, N_INPUTS, N_OUTPUTS, NEURONS,PARAMETERS):#
    # find best hyperparameters
    model = hyp_param_Optimize(train, N_INPUTS,N_OUTPUTS,NEURONS,PARAMETERS)

    history = [x for x in train]

    # walk-forward validation over each hour
    predictions = list()
    for i in range(len(test)):
        # predict the hour
        yhat_sequence = forecast(model, history, N_INPUTS)
        # store the prediction
        predictions.append(yhat_sequence)
        # get real observation and add to
        # history for predicting the next hour
        history.append(test[i, :])  # take each row of test with all values in columns
    # evaluate prediction minutes for each hour

    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :], predictions)

    return score, scores, predictions, history


##################################################################
#                              Main Part                          #
##################################################################

if __name__ == '__main__':
    np.random.seed(7)
    PATH = '../Power_Q_Phase.csv'
    df = pd.read_csv(PATH, header=0, parse_dates=[0], index_col=0)

    min_groups = resampledata_from_second_to_minutes(df)
    #plot_data(min_groups)
    train, test = split_dataset_(min_groups, N_OUTPUTS, TRAIN_NUMBERS, DATASET)

    score, scores, predictions, _ = evaluate_model(train, test, N_INPUTS, N_OUTPUTS, NEURONS,PARAMETERS)

    summerize_scores('LSTM Average', score, scores)
    print(predictions.shape)

    # evaluate model and get scores
    history = train[:, :].ravel()
    plt.plot(list(range(history.size)), history, '-', color='blue', label='history')
    # plt.plot(np.arange(history.size - 1, predictions.shape[0]), predictions[:, -1], marker='+', color='m', label='lstm')
    start_pos = history.size - 1
    end_pos = start_pos + predictions.shape[0] * predictions.shape[1]
    plt.plot(np.arange(start_pos, end_pos), test[:, :].ravel(), '-.', color='green', label='Actual')
    plt.plot(np.arange(start_pos, end_pos), predictions.ravel(), marker='+', color='m', label='Prediction')
    plt.legend()
    plt.show()

    minutes = range(N_OUTPUTS)
    minutes = list(minutes)
    plt.plot(minutes, scores, '-.', color='red', label='RMSE')
    plt.xlabel('Time')
    plt.ylabel('RMSE_Values')
    plt.legend(loc='upper right')
    plt.show()



