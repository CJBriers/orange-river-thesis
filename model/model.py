""" Create NN model """

import os
import util
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide messy TensorFlow warnings

def create_nn(input_dim, nr_units, nr_layers):
    """Compile NN model with real number as output"""
    print('Compiling NN model with {0} layers, {1} units, input dimension {2}...'.format(nr_layers, nr_units, input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(nr_units, activation='relu', input_shape=(input_dim,)))
    for _ in range(0, nr_layers - 1):

        # batch normalisation
        # model.add(keras.layers.Dense(nr_units, use_bias=False))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(nr_units, activation='relu')) # no batch normalisation
    model.add(keras.layers.Dropout(0.5)) # _dropout suffix
    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)

    # model.compile(loss='mean_squared_error', optimizer=sgd) # _mse suffix
    model.compile(loss='logcosh', optimizer=sgd) # _logcosh suffix
    # model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd) # _mslogerr
    print('Done compiling model')
    return model

def create_multi_cnn(input_dims):
    """Compile multi-input CNN model with real number as output"""
    print('Compiling multi-input CNN model, input dimensions {0}...'.format(input_dims))

    input_layers = []
    layers = []
    for input_dim in input_dims:
        input_layer = keras.layers.Input(shape=(input_dim, 1))
        input_layers.append(input_layer)
        layer = keras.layers.Conv1D(64, 3, strides=2, padding='valid', activation='relu')(input_layer)
        layer = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu')(layer)
        layer = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu')(layer)
        layer = keras.layers.Conv1D(256, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer)
        layer = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer)
        layer = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Flatten()(layer)
        layers.append(layer)

    output = keras.layers.concatenate(layers)
    output = keras.layers.Dense(2048, activation='relu')(output)
    output = keras.layers.Dense(1, activation='linear')(output)

    model = keras.models.Model(inputs=input_layers, outputs=output)
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)
    model.compile(loss='logcosh', optimizer=sgd)
    
    print('Done compiling model')
    return model

def create_multi_cnn_custom(input_dims):
    """Compile multi-input CNN model with real number as output"""
    print('Compiling custom multi-input CNN model, input dimensions {0}...'.format(input_dims))

    input_layers = []
    layers = []
    
    input_dim1 = input_dims[0]
    input_layer1 = keras.layers.Input(shape=(input_dim1, 1))
    input_layers.append(input_layer1)
    layer1 = keras.layers.Conv1D(64, 3, strides=2, padding='valid', activation='relu')(input_layer1)
    layer1 = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(256, 3, padding='same', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer1)
    layer1 = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer1)
    layer1 = keras.layers.Flatten()(layer1)
    layers.append(layer1)

    input_dim2 = input_dims[1]
    input_layer2 = keras.layers.Input(shape=(input_dim2, 2))
    input_layers.append(input_layer2)

    layer2 = keras.layers.Conv1D(64, 3, strides=2, padding='valid', activation='relu')(input_layer2)
    layer2 = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(256, 3, padding='same', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu')(layer2)
    layer2 = keras.layers.Conv1D(512, 3, padding='same', activation='relu')(layer2)
    layer2 = keras.layers.Flatten()(layer2)
    layers.append(layer2)

    output = keras.layers.concatenate(layers)
    output = keras.layers.Dense(2048, activation='relu')(output)
    output = keras.layers.Dense(1, activation='linear')(output)

    model = keras.models.Model(inputs=input_layers, outputs=output)
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)
    model.compile(loss='logcosh', optimizer=sgd)
    
    print('Done compiling model')
    return model

def create_multi_nn(input_dims):
    """Compile multi-input NN model with real number as output"""
    print('Compiling multi-input NN model, input dimensions {0}...'.format(input_dims))

    input_layers = []
    layers = []
    for input_dim in input_dims:
        input_layer = keras.layers.Input(shape=(input_dim,))
        input_layers.append(input_layer)
        layer = keras.layers.Dense(input_dim, activation='relu')(input_layer)
        layer = keras.layers.Dense(8192, activation='relu')(input_layer)
        layers.append(layer)

    output = keras.layers.concatenate(layers)
    output = keras.layers.Dense(1, activation='linear')(output)

    model = keras.models.Model(inputs=input_layers, outputs=output)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    print('Done compiling model')
    return model

def create_cnn(input_dim_1, input_dim_2, nr_layers, nr_units):
    """Compile CNN model with real number as output"""
    print('Compiling CNN model: {0} layers, input dimension {1}x{2}...'.format(nr_layers, input_dim_1, input_dim_2))
    model = keras.models.Sequential()

    if nr_layers > 0:
        model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(input_dim_1, input_dim_2)))
        model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    if nr_layers > 2:
        model.add(keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    if nr_layers > 4:
        model.add(keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(256, 3, padding='same', activation='relu'))
    if nr_layers > 6:
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    if nr_layers > 8:
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    #if nr_layers > 10:
    #    model.add(keras.layers.Conv1D(288, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(288, 3, padding='same', activation='relu'))
    #if nr_layers > 12:
    #    model.add(keras.layers.Conv1D(384, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(384, 3, padding='same', activation='relu'))
    
    model.add(keras.layers.Flatten())
    if nr_units > 0:
        model.add(keras.layers.Dense(nr_units, activation='relu'))
        print('Compiling CNN model: including fully connected layer with {0} units...'.format(nr_units))
        
    model.add(keras.layers.Dropout(0.5)) # _dropout suffix    
    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)

    # model.compile(loss='logcosh', optimizer=sgd)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd) # _mslogerr suffix
    print('Done compiling model')
    return model

def create_cnn_more_dropout(input_dim_1, input_dim_2, nr_layers, nr_units):
    """Compile CNN model with real number as output"""
    print('Compiling CNN model: {0} layers, input dimension {1}x{2}...'.format(nr_layers, input_dim_1, input_dim_2))
    model = keras.models.Sequential()

    if nr_layers > 0:
        model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(input_dim_1, input_dim_2)))
        model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    if nr_layers > 2:
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    if nr_layers > 4:
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(256, 3, padding='same', activation='relu'))
    if nr_layers > 6:
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    if nr_layers > 8:
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    #if nr_layers > 10:
    #    model.add(keras.layers.Conv1D(288, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(288, 3, padding='same', activation='relu'))
    #if nr_layers > 12:
    #    model.add(keras.layers.Conv1D(384, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(384, 3, padding='same', activation='relu'))
    
    model.add(keras.layers.Flatten())
    if nr_units > 0:
        model.add(keras.layers.Dense(nr_units, activation='relu'))
        print('Compiling CNN model: including fully connected layer with {0} units...'.format(nr_units))
        
    model.add(keras.layers.Dropout(0.2)) # _dropout suffix    
    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)

    # model.compile(loss='logcosh', optimizer=sgd)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd) # _mslogerr suffix
    print('Done compiling model')
    return model

def create_cnn_simple(input_dim_1, input_dim_2, nr_layers, nr_units):
    """Compile CNN model with real number as output"""
    print('Compiling CNN model: {0} layers, input dimension {1}x{2}...'.format(nr_layers, input_dim_1, input_dim_2))
    model = keras.models.Sequential()

    if nr_layers > 0:
        model.add(keras.layers.Conv1D(12, 3, padding='same', activation='relu', input_shape=(input_dim_1, input_dim_2)))
        model.add(keras.layers.Conv1D(12, 3, padding='same', activation='relu'))
    if nr_layers > 2:
        model.add(keras.layers.Conv1D(24, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(24, 3, padding='same', activation='relu'))
    if nr_layers > 4:
        model.add(keras.layers.Conv1D(48, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(48, 3, padding='same', activation='relu'))
    if nr_layers > 6:
        model.add(keras.layers.Conv1D(96, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(96, 3, padding='same', activation='relu'))
    if nr_layers > 8:
        model.add(keras.layers.Conv1D(96, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(96, 3, padding='same', activation='relu'))
    #if nr_layers > 10:
    #    model.add(keras.layers.Conv1D(288, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(288, 3, padding='same', activation='relu'))
    #if nr_layers > 12:
    #    model.add(keras.layers.Conv1D(384, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(384, 3, padding='same', activation='relu'))
    
    model.add(keras.layers.Flatten())
    if nr_units > 0:
        model.add(keras.layers.Dense(nr_units, activation='relu'))
        print('Compiling CNN model: including fully connected layer with {0} units...'.format(nr_units))
    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)

    model.compile(loss='logcosh', optimizer=sgd)
    print('Done compiling model')
    return model

def create_cnn_batch_norm(input_dim_1, input_dim_2, nr_layers, nr_units):
    """Compile CNN model (with batch norm) with real number as output"""
    print('Compiling CNN model: {0} layers, input dimension {1}x{2}...'.format(nr_layers, input_dim_1, input_dim_2))
    model = keras.models.Sequential()

    if nr_layers > 0:
        model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(input_dim_1, input_dim_2)))
        model.add(keras.layers.Conv1D(64, 3, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    if nr_layers > 2:
        model.add(keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(128, 3, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    if nr_layers > 4:
        model.add(keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(256, 3, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    if nr_layers > 6:
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    if nr_layers > 8:
        model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
        model.add(keras.layers.Conv1D(512, 3, padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    #if nr_layers > 10:
    #    model.add(keras.layers.Conv1D(288, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(288, 3, padding='same', activation='relu'))
    #if nr_layers > 12:
    #    model.add(keras.layers.Conv1D(384, 3, strides=2, padding='valid', activation='relu'))
    #    model.add(keras.layers.Conv1D(384, 3, padding='same', activation='relu'))
    
    model.add(keras.layers.Flatten())
    if nr_units > 0:
        model.add(keras.layers.Dense(nr_units, activation='relu'))
        print('Compiling CNN model: including fully connected layer with {0} units...'.format(nr_units))
    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=True)

    model.compile(loss='logcosh', optimizer=sgd)
    print('Done compiling model')
    return model

def create_bnn(input_dim_1, input_dim_2, nr_layers, nr_units):
    seedval = 3
    """Compile CNN model with real number as output"""
    print('Compiling CNN model: {0} layers, input dimension {1}x{2}...'.format(nr_layers, input_dim_1, input_dim_2))
    model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(input_dim_1, input_dim_2)))
    model.add(keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval)))

    model.add(keras.layers.Conv1D(128, 3, strides=2, padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval+1)))

    model.add(keras.layers.Conv1D(256, 3, strides=2, padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(256, 3, padding='same', activation='relu'))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval+2)))

    model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval+3)))

    model.add(keras.layers.Conv1D(512, 3, strides=2, padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(512, 3, padding='same', activation='relu'))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval+4)))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(nr_units, activation='relu'))
    print('Compiling CNN model: including fully connected layer with {0} units...'.format(nr_units))
    model.add(keras.layers.core.Lambda(lambda x: keras.backend.dropout(x, level=0.1, seed=seedval+5)))

    model.add(keras.layers.Dense(1, activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True)

    model.compile(loss='logcosh', optimizer=sgd)
    print('Done compiling model')
    return model

def create_rnn_lstm(input_dim_2):
    """Compile RNN-LSTM model with float as output"""
    print('Compiling RNN LSTM model with {0} units...'.format(input_dim_2))
    reg = keras.regularizers.L1L2(l1=0.01, l2=0.01)
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(input_dim_2, input_shape=(None, input_dim_2)))
    # model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    print('Done compiling model')
    return model

def create_rnn_gru(nr_units, nr_layers, input_dim_2):
    """Compile RNN-GRU model with float as output"""
    print('Compiling RNN GRU model with {0} units {1} channels and {2} unit final densely connected layer...'.format(nr_units, input_dim_2, nr_layers))
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(nr_units, input_shape=(None, input_dim_2)))
    if nr_layers > 1:
        model.add(keras.layers.Dense(nr_layers, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)
    print('Done compiling model')
    return model

def create_rnn_gru_dropout(nr_units, nr_layers, input_dim_2):
    """Compile RNN-GRU model with float as output"""
    print('Compiling RNN GRU model with {0} units {1} channels and {2} unit final densely connected layer...'.format(nr_units, input_dim_2, nr_layers))
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(nr_units, input_shape=(None, input_dim_2)))
    if nr_layers > 1:
        model.add(keras.layers.Dense(nr_layers, activation='relu'))
    model.add(keras.layers.Dropout(0.5)) # _dropout suffix   
    model.add(keras.layers.Dense(1, activation='linear'))

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='logcosh', optimizer=opt)
    print('Done compiling model')
    return model

def create_rnn_gru_variational_dropout(nr_units, nr_layers, input_dim_2):
    """Compile RNN-GRU model with float as output"""
    print('Compiling RNN GRU model with {0} units {1} channels and {2} unit final densely connected layer...'.format(nr_units, input_dim_2, nr_layers))
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(nr_units, input_shape=(None, input_dim_2), dropout=0.1, recurrent_dropout=0.1))
    if nr_layers > 1:
        model.add(keras.layers.Dense(nr_layers, activation='relu'))
    model.add(keras.layers.Dropout(0.5)) # _dropout suffix  
    model.add(keras.layers.Dense(1, activation='linear'))

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='logcosh', optimizer=opt)
    print('Done compiling model')
    return model


def create_rnn_gru_batchnorm(nr_units, nr_layers, input_dim_2):
    """Compile RNN-GRU model with float as output"""
    print('Compiling RNN GRU model with {0} units {1} channels and {2} unit final densely connected layer...'.format(nr_units, input_dim_2, nr_layers))
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(nr_units, input_shape=(None, input_dim_2)))
    if nr_layers > 1:
        model.add(keras.layers.Dense(nr_layers))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='logcosh', optimizer=opt)
    print('Done compiling model')
    return model

def create_nn_cat(input_dim, nr_units, nr_layers, nr_classes):
    """Compile NN model with categories as outputs"""
    print('Compiling NN model with {0} layers, {1} units, input dimension {2}...'.format(nr_layers, nr_units, input_dim))
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(nr_units, activation='relu', input_shape=(input_dim,)))
    #for _ in range(0, nr_layers - 1):
    filters = 32
    kernel_size = 24
    model.add(keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(nr_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print('Done compiling model')
    return model

def create(result_dir, input_dim, nr_layers, nr_units, network_type, input_dim_2):
    model_file = '{0}/model.h5'.format(result_dir)
    
    # load existing model
    model = util.load_model(model_file)
    if model is not None: 
        return model

    # TRY init='he_normal' weight initialisation

    if network_type == 'nn':
        model = create_nn(input_dim, nr_units, nr_layers)
    elif network_type == 'cnn':
        model = create_cnn(input_dim, input_dim_2, nr_layers, nr_units)
        # model = create_cnn_more_dropout(input_dim, input_dim_2, nr_layers, nr_units)
        # model = create_cnn_batch_norm(input_dim, input_dim_2, nr_layers, nr_units)
        # model = create_cnn_simple(input_dim, input_dim_2, nr_layers, nr_units)
    elif network_type == 'bnn':
        model = create_bnn(input_dim, input_dim_2, nr_layers, nr_units)
    elif network_type == 'multi_nn':
        model = create_multi_nn(input_dim)
    elif network_type == 'multi_cnn_custom':
        model = create_multi_cnn_custom(input_dim)
    elif network_type == 'multi_cnn':
        model = create_multi_cnn(input_dim)
        # elif network_type == 'nn_cat':
        #    model = create_nn_cat(input_dim, nr_units, nr_layers, cats)
    elif network_type == 'rnn_lstm':
        model = create_rnn_lstm(input_dim_2)
    elif network_type == 'rnn_gru':
        model = create_rnn_gru(nr_units, nr_layers, input_dim_2) # first run
        # model = create_rnn_gru_batchnorm(nr_units, nr_layers, input_dim_2) # second run
        # model = create_rnn_gru_dropout(nr_units, nr_layers, input_dim_2) # third run
        # model = create_rnn_gru_variational_dropout(nr_units, nr_layers, input_dim_2) # fourth run        
    
    return model
