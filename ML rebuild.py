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

def Get_Data_Label_Aux_train_Set(data_train, steps, t_tlag):
    cabinets = data_train
    stamps = data_train
    x_dim = len(cabinets)
    time_dim = len(stamps)
    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(steps, time_dim - steps):
        data_set.append(data_train[i - steps: i])

    for i in range(steps, time_dim - t_tlag):
        label_set.append(data_train[i:i + t_tlag])

    data_set = np.array(data_set)

    label_set = np.array(label_set)

    hour_set = np.array(hour_set)

    dayofweek_set = np.array(dayofweek_set)

    return data_set, label_set,hour_set, dayofweek_set


def Get_Data_Label_Aux_test_Set(data_test, steps, t_tlag):
    cabinets = data_test
    stamps = data_test
    x_dim = len(cabinets)
    time_dim = len(stamps)
    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []
    for i in range(time_dim - steps):
        data_set.append(data_test[i: i + steps])
       # label_set.append(data_test[i: i + t_tlag])

    for i in range( time_dim - t_tlag):
        label_set.append(data_test[i: i + t_tlag])



    data_set = np.array(data_set)
   # print("data set", data_set)
    label_set = np.array(label_set)
    #print("label set", label_set)
    hour_set = np.array(hour_set)

    dayofweek_set = np.array(dayofweek_set)

    return data_set, label_set,hour_set, dayofweek_set

def Split_train_Data(X_full, Y_full):
    n = Y_full.shape[0]-12
    indices = np.arange(n)
    RS = RandomState(1024)
    RS.shuffle(indices)
    sep_1 = int(float(n))
    train_indices = indices[:sep_1]

    X_train = X_full[train_indices]
    Y_train = Y_full[train_indices]


    return X_train, Y_train

def Split_test_Data(X_full, Y_full):
    n = Y_full.shape[0]-12
    indices = np.arange(n)
   # RS = RandomState(1024)
   # RS.shuffle(indices)
    sep_1 = int(float(n))

    test_indices = indices[:sep_1]

    X_test = X_full[test_indices]
    Y_test = Y_full[test_indices]


    return X_test, Y_test

if __name__ == "__main__":
    #######################################################
    # load 2015 speed data
    #######################################################

    speedMatrix = pd.read_pickle('H:\\Paper Code\\Speed data\\speed_matrix_2015')

    loopgroups_full = speedMatrix.columns.values


    data_train= speedMatrix.iloc[:94608 :].values

    data_test= speedMatrix.iloc[94608 :].values

    time_lag = 12
    t_time_lag = 6
    print('time lag :', time_lag)

  #  X_full, Y_full, hour_full, dayofweek_full = Get_Data_Label_Aux_train_Set(data_train, time_lag, t_time_lag)

   # print('X_full train shape: ', X_full.shape, 'Y_full shape:', Y_full.shape)
  #  X_train, Y_train  = Split_train_Data(X_full, Y_full)

    X_test_full, Y_test_full, hour_full, dayofweek_full = Get_Data_Label_Aux_test_Set(data_test, time_lag, t_time_lag)
    X_test, Y_test = Split_test_Data(X_test_full, Y_test_full)

  #  print('X_train shape: ', X_train.shape, 'Y_train shape:', Y_train.shape)
    print('X_test shape: ', X_test.shape, 'Y_test shape:', Y_test.shape)

    #######################################################
    # bound training data to 0 to 100
    # get the max value of X to scale X
    #######################################################
  #  X_train = np.clip(X_train, 0, 100)
    X_test = np.clip(X_test, 0, 100)
    X_max= 100

  ##  X_max = np.max([np.max(X_train), np.max(X_test)])
  #  X_min = np.min([np.min(X_train), np.min(X_test)])
   # print('X_full max:', X_max)

    #######################################################
    # scale data into 0~1
    #######################################################
  #  X_train_scale = X_train / X_max
    X_test_scale = X_test / X_max

  #  Y_train_scale = Y_train / X_max
    Y_test_scale = Y_test / X_max

    model_epoch = 1
    patience = 20

    print("#######################################################")
    print("model_FCN-Bi_LSTMatt")
    print("time_lag", time_lag)

    from keras.models import load_model
    from keras.utils import CustomObjectScope
    from BDLSTM_FCN import Attention

    model = load_model('TL12-LSTM for TTL-6- 70ep_tl12.h5', custom_objects={'Attention': Attention})
  #  model = load_model('TL12-BDLSTM1 for TTL-6- 96ep_tl12.h5')
    model.summary()
    Y_pred_test = model.predict(X_test_scale)
    y_true = Y_test_scale

    Y_pred_test= Y_pred_test *X_max
    y_true= y_true * X_max




    # Evaluation metrics
    vs = metrics.explained_variance_score(y_true[:,0,:], Y_pred_test[:,0,:])
    mae = metrics.mean_absolute_error(y_true[:,0,:], Y_pred_test[:,0,:])

    mse = metrics.mean_squared_error(y_true[:,0,:], Y_pred_test[:,0,:])
    r2 = metrics.r2_score(y_true[:,0,:], Y_pred_test[:,0,:])
    mape = np.mean(np.abs((y_true[:,0,:]- Y_pred_test[:,0,:]) / y_true[:,0,:])) * 100



    print('Explained_various_Score: %f' % vs)
    print('MAE : %f' % mae)
    print('MAPE:%f' % mape)
    print('MSE : %f' % mse)
    print('RMSE : %f' % math.sqrt(mse))
    print('r2: %f' % r2)
   # print('epoch %f' % epochs)


   # print(Y_pred_test.head())

    y_true = [item[0] for item in y_true]
    y_true= [item[204] for item in y_true]
    y_true = np.array(y_true)
    Y_pred_test = [item[0] for item in Y_pred_test]
    Y_pred_test = [item[204] for item in Y_pred_test]
    Y_pred_test= np.array(Y_pred_test)

####Plotting code#######

    plt.figure(figsize=(16, 12))
    plt.plot(y_true[136:424], 'b', label= 'Truth Speed' )
    plt.plot(Y_pred_test[136:424], 'r', marker='.', label= 'Predicted Speed' )
   # plt.plot(df['pred'][:, 0, :], 'r')
  #  plt.tick_params(left=False, labelleft=True,)
  #  plt.tight_layout()
  #  sns.despine(top= True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Speed(mph)', size=15)
    plt.xlabel('Time Steps(5 Minutes)', size=15)
    plt.legend(fontsize=15)
    plt.show()
    print("thanks")
    plt.save()

    predicted = Y_pred_test / X_max

    Y_pred_test = []
    random_day = random.randint(0, 6)
    idx = random_day * 100
    Y_pred_test.append(predicted[idx:idx + 100])
   # plot_results(random_day, Y_test_scale[idx:idx + 288], Y_pred_test)