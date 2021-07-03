import BDLSTM_FCN
import numpy as np
import datetime
import pandas as pd
from numpy.random.mtrand import RandomState
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split

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
        dayofweek = datetime.datetime.strptime(stamp[0:10], '%Y-%M-%d').strftime('%w')
        dayofweek_set.append(float(dayofweek))

    data_set = np.array(data_set)
   # print('data set', data_set)
    label_set = np.array(label_set)
  #  print('laebl set', label_set)
    hour_set = np.array(hour_set)
   # print('hour set', hour_set)
    dayofweek_set = np.array(dayofweek_set)
  #  print('day of week ', dayofweek)
    return data_set, label_set,hour_set, dayofweek_set


def SplitData(X_test,Y_test, t_time_lag):
    n = Y_test.shape[0]
    indices = np.arange(n)

    ##############################################
    # For future multi-time steps
    test_indices = []



    for i in range(n - t_time_lag):
       # test_indices.append(speedMatrix[i: i + t_time_lag])
        X_test.append(X_test[i: i + t_time_lag])
        Y_test

   # print("indices", test_indices)
    data_set = np.array(X_test)
    test_indices = data_set
   # print("test incides : ", test_indices)

  ####################################################


    X_test = X_full[test_indices]

    Y_test = Y_full[test_indices]

    return  X_test, Y_test




if __name__ == "__main__":
    #######################################################
    # load 2015 speed data
    #######################################################

    speedMatrix = pd.read_pickle('H:\\Paper Code\\Speed data\\speed_matrix_2015')
    print('speedMatrix shape:', speedMatrix.shape)
    loopgroups_full = speedMatrix.columns.values
    print(speedMatrix)

    time_lag = 1
    t_time_lag = 6
    print('time lag :', time_lag)



    X_full, Y_full, hour_full, dayofweek_full = Get_Data_Label_Aux_Set(speedMatrix, time_lag, )
    print('X_full shape: ', X_full.shape, 'Y_full shape:', Y_full.shape)

    #######################################################
    # split full dataset into training, validation and test dataset

    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full,  test_size = 0.1, random_state = 1042)

   ######################################################################################


################################################################


    print('X-train', X_train)

    print('Y-train', Y_train)
    print('X-test', X_test)
    print('Y-test', Y_test)
   # X_test,Y_test = SplitData(X_test,Y_test, t_time_lag)





    #######################################################


    print('X_train shape: ', X_train.shape, 'Y_train shape:', Y_train.shape)
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

    model_epoch = 500
    patience = 20

    print("#######################################################")
    print("model_FCN-Bi_LSTMatt")
    print("time_lag", time_lag)

    from keras.models import load_model
    from keras.utils import CustomObjectScope
    from BDLSTM_FCN import Attention
    model = load_model('H:\\Paper Code\\test6___-mse-rmse1ep_tl1.h5', custom_objects={'Attention': Attention})
    model.summary()
    X_test_scale.flatten()

    Y_pred_test = model.predict(X_test_scale)

    y_true = Y_test_scale

# Evaluation metrics
vs = metrics.explained_variance_score(y_true, Y_pred_test)
mae = metrics.mean_absolute_error(y_true, Y_pred_test)
mse = metrics.mean_squared_error(y_true, Y_pred_test)
r2 = metrics.r2_score(y_true, Y_pred_test)
mape = np.mean(np.abs((y_true - Y_pred_test) / y_true)) * 100

print('Explained_various_Score: %f' % vs)
print('MAE : %f' % mae)
print('MAPE:%f' % mape)
print('MSE : %f' % mse)
print('RMSE : %f' % math.sqrt(mse))
print('r2: %f' % r2)
# print('epoch %f' % epochs)