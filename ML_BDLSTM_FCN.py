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
    BDLSTM_output = Attention(12)(BDLSTM_output)

    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, activation='sigmoid', padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)
    combine_output = concatenate([FCN_output, BDLSTM_output])
    combine_output=  RepeatVector(6)(combine_output)
    final_output = Dense(323,activation='sigmoid')(combine_output)
    NAME = "TL12-Proposed Model for TTL-6-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])
    return final_model, history

def LSTMS(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()
    model.add(LSTM(output_dim=X_train.shape[2], return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(RepeatVector(12))

    NAME = "TL3-LSTMs for TTL-12-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X_train, Y_train, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history
def LSTMDNN(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()


    model.add(LSTM(output_dim=X_train.shape[2], return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(X_train.shape[2]))
    model.add(RepeatVector(3))

    NAME = "TL3-CheckLSTM-DNN for TTL-3{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X_train, Y_train, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def GRU(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    model = Sequential()

    model.add(keras.layers.GRU(X_train.shape[2], return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(RepeatVector(12))
    #model.add(Dense(X.shape[2]))
    NAME = "TL3-GRU for TTL-12-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X_train, Y_train, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history


def FCN(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='speed')
    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)
    FCN_output = RepeatVector(12)(FCN_output)

    final_output = Dense(323,activation='sigmoid')(FCN_output)

    NAME = "TL12-FCN for TTL-12-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def LSTMS_FCN(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X_train.shape[1], X_train.shape[2]), output_dim = X_train.shape[2], return_sequences=False)(speed_input)

    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)

    combine_output = concatenate([main_output, FCN_output])
    combine_output = RepeatVector(12)(combine_output)
    final_output = Dense(323,activation='sigmoid')(combine_output)

    NAME = "TL3-LSTM-FCN for TTL-12-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def BDLSTMS(X_train, Y_train, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X_train.shape[1], X_train.shape[2]), output_dim = X_train.shape[2], return_sequences=False), merge_mode='ave')(speed_input)

    final_output=  RepeatVector(6)(main_output)

    NAME = "TL12-BDLSTM for TTL-6-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])
    return final_model, history