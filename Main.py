import BDLSTM_FCN
import numpy as np
import datetime
import pandas as pd
from numpy.random.mtrand import RandomState
import matplotlib as mpl
import matplotlib.pyplot as plt
import  sklearn.metrics as metrics
import math
import random

def plot_results(day, y_true, Y_pred_test):
    day = 22 + day
    d = '2019-10-' + str(day) + ' 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for y_pred in Y_pred_test:
        ax.plot(x, y_pred, label='BDLSTM-FCN')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def Get_Data_Label_Aux_Set(speedMatrix, steps):
    cabinets = speedMatrix.columns.values
    stamps = speedMatrix.index.values
    x_dim = len(cabinets)
    time_dim = len(stamps)

    speedMatrix = speedMatrix.iloc[:, :].values

    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(time_dim - steps):
        data_set.append(speedMatrix[i: i + steps])
        label_set.append(speedMatrix[i + steps])
        stamp = stamps[i + steps]
        hour_set.append(float(stamp[11:13]))
        dayofweek = datetime.datetime.strptime(stamp[0:6], '%Y-%M-%d').strftime('%w')
        dayofweek_set.append(float(dayofweek))

    data_set = np.array(data_set)
    label_set = np.array(label_set)
    hour_set = np.array(hour_set)
    dayofweek_set = np.array(dayofweek_set)
    return data_set, label_set, hour_set, dayofweek_set


def SplitData(X_full, Y_full, hour_full, dayofweek_full, train_prop=0.7, valid_prop=0.2, test_prop=0.1):
    n = Y_full.shape[0]
    indices = np.arange(n)
    RS = RandomState(1024)
    RS.shuffle(indices)
    sep_1 = int(float(n) * train_prop)
    sep_2 = int(float(n) * (train_prop + valid_prop))
    print('train : valid : test = ', train_prop, valid_prop, test_prop)
    train_indices = indices[:sep_1]
    valid_indices = indices[sep_1:sep_2]
    test_indices = indices[sep_2:]
    X_train = X_full[train_indices]
    X_valid = X_full[valid_indices]
    X_test = X_full[test_indices]
    Y_train = Y_full[train_indices]
    Y_valid = Y_full[valid_indices]
    Y_test = Y_full[test_indices]
    #hour_train = hour_full[train_indices]
   # hour_valid = hour_full[valid_indices]
    #hour_test = hour_full[test_indices]
    #dayofweek_train = dayofweek_full[train_indices]
    #dayofweek_valid = dayofweek_full[valid_indices]
   # dayofweek_test = dayofweek_full[test_indices]
    return X_train, X_valid, X_test, \
           Y_train, Y_valid, Y_test
          # hour_train, hour_valid, hour_test, \
           #dayofweek_train, dayofweek_valid, dayofweek_test


def MeasurePerformance(Y_test_scale, Y_pred, X_max, model_name='default', epochs=30, model_time_lag=10):
    time_num = Y_test_scale.shape[0]
    loop_num = Y_test_scale.shape[1]

    difference_sum = np.zeros(time_num)
    diff_frac_sum = np.zeros(time_num)

    for loop_idx in range(loop_num):
        true_speed = Y_test_scale[:, loop_idx] * X_max
        predicted_speed = Y_pred[:, loop_idx] * X_max
        diff = np.abs(true_speed - predicted_speed)
        diff_frac = diff / true_speed
        difference_sum += diff
        diff_frac_sum += diff_frac

    difference_avg = difference_sum / loop_num
    MAPE = diff_frac_sum / loop_num * 100

    print('MAE :', round(np.mean(difference_avg), 3), 'MAPE :', round(np.mean(MAPE), 3), 'STD of MAE:',round(np.std(difference_avg), 3), 'MSE :')
    print('Epoch : ', epochs)


if __name__ == "__main__":
    #######################################################
    # load 2015 speed data
    #######################################################

    speedMatrix = pd.read_pickle('H:\\Paper Code\\Speed data\\speed_matrix_2015')
    print('speedMatrix shape:', speedMatrix.shape)
    loopgroups_full = speedMatrix.columns.values
    print(speedMatrix)
    time_lag = 10
    print('time lag :', time_lag)

    X_full, Y_full = Get_Data_Label_Aux_Set(speedMatrix, time_lag)
    print('X_full shape: ', X_full.shape, 'Y_full shape:', Y_full.shape)

    #######################################################
    # split full dataset into training, validation and test dataset
    #######################################################
    X_train, X_valid, X_test, \
    Y_train, Y_valid, Y_test = SplitData(X_full, Y_full, hour_full=0, dayofweek_full=0, train_prop=0.9, valid_prop=0.0, test_prop=0.1)
   # hour_train, hour_valid, hour_test, \
   # dayofweek_train, dayofweek_valid, dayofweek_test \
      #  = SplitData(X_full, Y_full, hour_full, dayofweek_full, train_prop=0.9, valid_prop=0.0, test_prop=0.1)
    print('X_train shape: ', X_train.shape, 'Y_train shape:', Y_train.shape)
    print('X_valid shape: ', X_valid.shape, 'Y_valid shape:', Y_valid.shape)
    print('X_test shape: ', X_test.shape, 'Y_test shape:', Y_test.shape)

    #######################################################
    # bound training data to 0 to 100
    # get the max value of X to scale X
    #######################################################
    X_train = np.clip(X_train, 0, 100)
    X_test = np.clip(X_test, 0, 100)

    X_max = np.max([np.max(X_train), np.max(X_test)])
    X_min = np.min([np.min(X_train), np.min(X_test)])
    print('X_full max:', X_max)

    #######################################################
    # scale data into 0~1
    #######################################################
    X_train_scale = X_train / X_max
    X_test_scale = X_test / X_max

    Y_train_scale = Y_train / X_max
    Y_test_scale = Y_test / X_max

    model_epoch = 200
    patience = 20

    print("#######################################################")
    print("model_Bi_LSTMatt")
    print("time_lag", time_lag)

    model_2_Bi_LSTM, history_2_Bi_LSTM = BDLSTM_FCN.train_2_Bi_LSTM(X_train_scale, Y_train_scale, epochs=model_epoch)
    model_2_Bi_LSTM.save('final_modelSBULSTMTL10WH' + str(len(history_2_Bi_LSTM.losses)) + 'ep' + '_tl' + str(time_lag) + '.h5')
    Y_pred_test = model_2_Bi_LSTM.predict(X_test_scale)
    MeasurePerformance(Y_test_scale, Y_pred_test, X_max, model_name='default', epochs=len(history_2_Bi_LSTM.losses),
                       model_time_lag=10)
