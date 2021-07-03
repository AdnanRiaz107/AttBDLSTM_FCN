from keras.callbacks import Callback
from keras.callbacks import EarlyStopping ,TensorBoard
from keras.layers import *
np.random.seed(1024)
from keras.layers import *
from keras.models import *
from keras import backend as K
#from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from time import time
import time
from keras.layers import Layer
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class Attention(Layer):
    def __init__(self, step_dim= None , bias=True,W_regularizer=None, b_regularizer=None,W_constraint=None, b_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

        super(Attention, self).build(input_shape) #be sure to call it at the end

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))


        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


    def get_config(self):
        config = {
            'step_dim': self.step_dim
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))



import keras
kullback_leibler_divergence = keras.losses.kullback_leibler_divergence
K = keras.backend
#In this example, 0.01 is the regularization weight, and 0.05 is the sparsity target.
def kl_divergence_regularizer(inputs):
        means = K.mean(inputs, axis=0)
        return 0.01 * (kullback_leibler_divergence(0.05, means)
                       + kullback_leibler_divergence(1 - 0.05, 1 - means))

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
#BDLSTM with FCN for future  multi step horizon
def BDLSTMS_FCN(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X_train.shape[1], X_train.shape[2]), activation='sigmoid', output_dim = X_train.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional(LSTM(input_shape = (X_train.shape[1], X_train.shape[2]),activation='sigmoid', output_dim = X_train.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(3)(BDLSTM_output)

    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, activation='sigmoid', padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)

    combine_output = concatenate([FCN_output, BDLSTM_output])
    combine_output=  RepeatVector(6)(combine_output)

    final_output = Dense(323,activation='sigmoid')(combine_output)
    NAME = "TL3-Proposed Model for TTL-6-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])


    return final_model, history






def BDLSTMS(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), activation='sigmoid', output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]),activation='sigmoid', output_dim = X.shape[2], return_sequences=False), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(1)(BDLSTM_output)

    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, activation='sigmoid', padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)

    FCN_output = GlobalAveragePooling1D()(y)

    combine_output = concatenate([FCN_output, BDLSTM_output])



    final_output = Dense(323,activation='sigmoid')(combine_output)
    NAME = "BDLSTMS_FCN__MSE_RMSE_TL_3A-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    History= final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])
    import matplotlib.pyplot as plt
    plt.plot(History.history['loss'])
    plt.xlabel('epoch', fontsize=16)

    plt.ylabel('loss', fontsize=16)

    plt.show()
    plt.plot(History.history['loss'])

    plt.plot(History.history['val_loss'])
    plt.legend(['train', 'validation'], loc='upper right', fontsize='large')
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)

    plt.show()
    return final_model, history



def LSTMs(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    print('X.shape 1   ', X.shape[1])
    print('X.shape 2   ', X.shape[2])

    main_output = LSTM(100, activation='sigmoid', return_seq=True)(speed_input)
    main_output2 = Dense(323, activation= 'sigmoid' )(main_output)
   # main_output= Attention(3,323)(main_output)
    NAME = "....".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[main_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def abc(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    main_output_1 = LSTM(100, activation='sigmoid')(speed_input)
    main_output_2 = Dense(1, activation='sigmoid')(main_output_1)
    final_model = Model(input=[speed_input], output=[main_output_2])


   # final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    history = final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs,
                                    callbacks=[history, earlyStopping])
    return final_model, history


def abcd(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):

    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    LSTM_1 = LSTM(100, activation='tanh')(speed_input)
    Repeat_Vector_1 = RepeatVector(3)(LSTM_1)
    DropOut_1 = Dropout(0.4)(Repeat_Vector_1)
    Batch_Normalization_1 = BatchNormalization()(DropOut_1)
    LSTM_2 = LSTM(100, activation='tanh', return_sequences=True)(Batch_Normalization_1)
    TimeDistriButed_1 = TimeDistributed(Dense(100, activation='tanh'))(LSTM_2)
    DropOut_2 = Dropout(0.4)(TimeDistriButed_1)
    Batch_Normalization_2 = BatchNormalization()(DropOut_2)
    TimeDistriButed_2 = TimeDistributed(Dense(323))(Batch_Normalization_2)
    model = Model(inputs=[speed_input], outputs=[TimeDistriButed_2])

    NAME = "..._tL_(r)-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,  write_grads=True)
    model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    history =model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])


    return model, history


def LSTMS_FCN(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    print('x-shape[1]', X.shape[1])
    print('x-shape[2]', X.shape[2])
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=False)(speed_input)
   # main_output =LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=False) (main_output)


    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)

    FCN_output = GlobalAveragePooling1D()(y)

    combine_output = concatenate([main_output, FCN_output])

    final_output = Dense(323,activation='sigmoid')(combine_output)


    NAME = "..._tL_(r)-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def LSTMS_DNN(X, Y, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()

    model.add(LSTM(output_dim=X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(X.shape[2]))
    NAME = "LSTM-DNN_TL_15-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def LSTMS(X, Y, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()

    model.add(LSTM(output_dim=X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))

    NAME = "LSTM-_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def GRU(X, Y, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()

    model.add(keras.layers.GRU(X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dense(X.shape[2]))
    NAME = "GRU_TL_15-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def BDLSTMS_deepA(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    encoded = Dense(128, activation='relu')(speed_input)
    encoded = Dense(64, activation='relu')(encoded)
    #  sparsity concept  activity_regularizer=regularizers.l1(10e-5)// KL divergence
    encoded = Dense(32, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(32, activation='sigmoid')(decoded)

    main_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(decoded)
    BDLSTM_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(10)(BDLSTM_output)

    #y = Permute((2, 1))(BDLSTM_output)
   # print(y)
   # y = Conv1D(128, 1, padding='same', kernel_initializer='he_uniform')(y)
   # y = BatchNormalization()(y)
   # y = Activation('sigmoid')(y)

    NAME = "2attBDLSTMS_deepA".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[BDLSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history



def FCN(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)

    final_output = Dense(323,activation='sigmoid')(FCN_output)

    NAME = "FCN(testTL5)-TL_6--Frmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history



def FNN(X, Y, epochs=30, validation_split=0.2, patience=10):
    model = Sequential()
    n_steps = 3
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')


    # flatten input
    n_input = X.shape[1] , X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    #x_input = speed_input.reshape((1, n_steps))
    model.add(Dense(323, activation='sigmoid', input_dim=n_steps))
    model.add(Dense(1,activation='sigmoid' ))

    NAME = "FNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,write_grads=True)

    model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.xlabel('epoch', fontsize=16)
    # plt.set_xlim(bottom=0)
    # plt.xlim(left=0)#, right)
    plt.ylabel('loss', fontsize=16)
    plt.savefig("./resulted_plotes/train_loss.jpg")
    plt.show()
    plt.plot(history.history['loss'])


    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'], loc='upper right', fontsize='large')
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.savefig("./resulted_plotes/all_loss.jpg")
    plt.show()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.legend(['train', 'validation'], loc='lower right', fontsize='large')
    plt.savefig("./resulted_plotes/_accuracy.jpg")
    plt.show()
    plt.plot(history.history['lr'])
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('learning rate', fontsize=16)
    plt.savefig("./resulted_plotes/learning_rate.jpg")
    plt.show()




    return model, history

def LSTMS_GRU(X, Y, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()

    model.add(LSTM(output_dim=X.shape[2], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.GRU(X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    NAME = "LSTM-GRU_TL_15-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history


def FNN_test(X, Y, epochs=30, validation_split=0.2, patience=10):
    model = Sequential()
    n_steps = 3
    # flatten input
    x_input = X.reshape((1, n_steps))
    model.add(Dense(323, activation='sigmoid', input_dim=n_steps))
    model.add(Dense(1,activation='sigmoid' ))

    NAME = "FNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,write_grads=True)

    model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history


def train_2_Bi_LSTM(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
   # lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(lstm_output)
    lstm_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2],return_sequences=True)(lstm_output)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2])(lstm_output)

    NAME = "timelag_10_SBULSTM(Lstm wWH)_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)

    final_model = Model(input=[speed_input], output=[main_output])

    final_model.summary()

    final_model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def Feed(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    model = Sequential()

    model.add(Dense(units=500, input_shape=(X.shape[1], X.shape[2]),activation="relu",kernel_initializer="random_uniform",bias_initializer="zeros"))
    model.add(Dense(units=323, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
    NAME = "FNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,write_grads=True)
    model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def BiLSTM(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=False), merge_mode='ave')(speed_input)

    NAME = "timelag_12_BDLSTM_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True)

    final_model = Model(input=[speed_input], output=[lstm_output])

    final_model.summary()

    final_model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def Feed(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    model = Sequential()

    model.add(Dense(units=500, input_shape=(X.shape[1], X.shape[2]),activation="relu",kernel_initializer="random_uniform",bias_initializer="zeros"))
    model.add(Dense(units=323, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
    NAME = "FNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,write_grads=True)
    model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history