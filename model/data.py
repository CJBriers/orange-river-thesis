""" Construct dataset """

import math
import pandas as pd
import numpy as np
import keras
import csv

def one_hot_encode_object_array(arr, nr_classes):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    _, ids = np.unique(arr, return_inverse=True)
    return keras.utils.to_categorical(ids, nr_classes)

def produce_diff(model, result_dir, down_station, input_list, include_time, sample_size, network_type, denormalise, roundit):
    
    # include_diff always false, we don't need to prodice a diff for models that include a diff in the input
    (y_train, x_train, y_cv, x_cv, y_test, x_test, _, _, train_y_max, train_y_min, train_idx, test_idx, cv_idx, _, _) = construct(down_station, input_list, include_time, sample_size, network_type)

    y_train_pred = model.predict(x_train)
    y_train_pred = y_train_pred.ravel()
    y_cv_pred = model.predict(x_cv)
    y_cv_pred = y_cv_pred.ravel()
    y_test_pred = model.predict(x_test)
    y_test_pred = y_test_pred.ravel()
    
    pred = np.concatenate((y_train_pred, y_test_pred, y_cv_pred))
    real = np.concatenate((y_train, y_test, y_cv))
    idx = np.concatenate((train_idx, test_idx, cv_idx))
    
    if denormalise:
        pred = pred * (train_y_max - train_y_min) + train_y_min
        real = real * (train_y_max - train_y_min) + train_y_min

    if roundit:
        pred = np.rint(pred)
        real = np.rint(real)

    pred_file = '{0}/{1}-pred-{2}.txt'.format(result_dir, down_station, network_type)
    np.savetxt(pred_file, np.transpose((idx, pred)), delimiter=',', fmt="%s")
    
    real_file = '{0}/{1}-real-{2}.txt'.format(result_dir, down_station, network_type)
    np.savetxt(real_file, np.transpose((idx, real)), delimiter=',', fmt="%s")

    diff_file = open('{0}/{1}-diff-{2}.txt'.format(result_dir, down_station, network_type), 'w+')
    np.savetxt(diff_file, np.transpose((idx, np.subtract(real, pred))), delimiter=',', fmt="%s")

# input_list is a tuple (input_name, interpolation_method, shift_offset, trim_nan)
def construct(down_station, input_list, include_time, sample_size, network_type):
    """Construct training dataset"""

    time_resolution = '1H'
    print('Constructing training data with resolution {0}'.format(time_resolution))
    
    ################################################################################################
    # downstream station

    target = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(down_station), parse_dates=['Date'])
    target = target.set_index(['Date'])
    target.index = pd.to_datetime(target.index)
    target_mean = target.resample(time_resolution).mean()
    first_date = target_mean.index.values[0]
    last_date = target_mean.index.values[-1]
    target_mean = target_mean.interpolate(method='linear')
    target_count = target.resample(time_resolution).count()
    target_count = target_count.values.astype(int)[:, 0]
    target_count = np.convolve(target_count, np.ones(24, dtype=int), 'full') # ignore training sample if more than day's worth of data is missing
    print('Downstream data for {0} downsampled'.format(down_station))

    ################################################################################################
    # input stations

    inputs_mean = []
    inputs_count = []
    for input_idx in range(0, len(input_list)):
        input_name = input_list[input_idx][0]
        input_interpolation_method = input_list[input_idx][1]
        input_shift_offset = input_list[input_idx][2]
        input_data = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(input_name), parse_dates=['Date'])
        input_data = input_data.set_index(['Date'])
        input_data.index = pd.to_datetime(input_data.index)
        if input_interpolation_method == 'linear':
            input_data_mean = input_data.resample(time_resolution).mean()
        else:
            input_data_mean = input_data.resample(time_resolution).pad()
        if input_shift_offset > 0:
            input_data_mean = input_data_mean.shift(sample_size)
        input_data_first_date = input_data_mean.index.values[0]
        if input_data_first_date > first_date:
            first_date = input_data_first_date
        input_data_last_date = input_data_mean.index.values[-1]
        if input_data_last_date < last_date:
            last_date = input_data_last_date
        input_data_mean = input_data_mean.interpolate(method=input_interpolation_method)
        input_data_count = input_data.resample(time_resolution).count()
        input_data_count = input_data_count.values.astype(int)[:, 0]
        input_data_count = np.convolve(input_data_count, np.ones(24, dtype=int), 'full') # ignore training sample if more than day's worth of data is missing
        inputs_mean.append(input_data_mean)
        inputs_count.append(input_data_count)
        print('Input for {0} downsampled using {1} interpolation and shift of {2} timesteps'.format(input_name, input_interpolation_method, input_shift_offset))

    ################################################################################################
    # trim input and output arrays to equal lengths

    print('Data before {0} ignored'.format(first_date))
    print('Data after {0} ignored'.format(last_date))
    
    lower_idx = 0
    upper_idx = 0

    for input_idx in range(0, len(inputs_mean)):
        lower_idx = inputs_mean[input_idx].index.get_loc(first_date)
        upper_idx = inputs_mean[input_idx].index.get_loc(last_date)
        inputs_mean[input_idx] = inputs_mean[input_idx].head(upper_idx).tail(upper_idx - lower_idx)
        inputs_count[input_idx] = inputs_count[input_idx][sample_size+lower_idx:upper_idx]

    print('{0} training samples before trimming dates'.format(target_mean.index.size))

    lower_idx = target_mean.index.get_loc(first_date)
    upper_idx = target_mean.index.get_loc(last_date)
    target_mean = target_mean.head(upper_idx).tail(upper_idx - lower_idx - sample_size)
    
    print('{0} training samples after trimming dates'.format(target_mean.index.size))

    target_count = target_count[sample_size+lower_idx:upper_idx]
    train = target_mean.copy(deep=True)

    ################################################################################################
    # Add time sine and cosine values

    time_inputs = 0
    if include_time:
        print('Including time of year as input')
        day_of_year = target_mean.index.to_series().apply(lambda x: x.timetuple().tm_yday/365.25 * 2 * math.pi)
        # day_of_year = target_mean.index.to_series().apply(lambda x: (x.timetuple().tm_yday - 1)/365.25 * 2 * math.pi) # this is more correct
        time_sin = day_of_year.apply(np.sin)
        time_cos = day_of_year.apply(np.cos)
        if (network_type == 'bnn') or (network_type == 'cnn') or (network_type == 'multi_cnn') or (network_type == 'multi_cnn_custom') or (network_type == 'rnn_lstm') or (network_type == 'rnn_gru'):
            for i in range(0, sample_size):
                train['TimeSin{0}'.format(i)] = time_sin
                train['TimeCos{0}'.format(i)] = time_cos
        else:
            train['TimeSin'] = time_sin
            train['TimeCos'] = time_cos
        time_inputs = 2

    ################################################################################################
    # Add inputs to training dataset

    for input_idx in range(0, len(input_list)):
        input_name = input_list[input_idx][0]
        for i in range(0, sample_size):
            shifted = inputs_mean[input_idx].shift(sample_size-i)
            train['Inflow_{0}_{1}'.format(input_name, i)] = shifted

    # Last column should be target values
    train = train.assign(Target=target_mean.values.astype(float))

    #############################################################################################################
    # Nuke samples where too much data is missing

    mask = target_count.astype(float)
    mask[mask == 0] = 'nan'
    
    # uncomment if you want to trim based on high/low flow regime
    # mask_y = train.values[:, -1]
    # mask[mask_y > 200] = 'nan'

    for input_idx in range(0, len(input_list)):
        input_name = input_list[input_idx][0]
        trim_nan = input_list[input_idx][3]
        if trim_nan:
            mask_input = inputs_count[input_idx]
            mask[mask_input == 0] = 'nan'
            print('Removing nan values for {0}'.format(input_name))
        else:
            print('Not removing nan values for {0}'.format(input_name))

    # creating copy for running full prediction on
    traincopy = train.copy()

    print('{0} training samples before trimming missing values'.format(train.index.size))
    train = train.assign(Mask=mask)
    train = train.dropna(axis=0, subset=train.columns[:])
    print('{0} training samples after trimming missing values'.format(train.index.size))

    #############################################################################################################
    # Clean up unused columns

    train = train.drop('Value', 1)
    train = train.drop('Mask', 1)

    traincopy = traincopy.drop('Value', 1)

    #############################################################################################################
    # Split training, CV and test datasets

    idx_test = int(0.6 * len(train))
    idx_cv = int(0.8 * len(train))
    cv = train.tail(len(train) - idx_cv)
    test = train.head(idx_cv).tail(idx_cv - idx_test)
    train = train.head(idx_test)

    #############################################################################################################
    # Split input and output 

    full_first_date = pd.to_datetime("2011-08-01 00:00:00")
    full_last_date = pd.to_datetime("2014-07-08 00:00:00")
    full_lower_idx = traincopy.index.get_loc(full_first_date, method='nearest')
    full_upper_idx = traincopy.index.get_loc(full_last_date, method='nearest')
    traincopy = traincopy.head(full_upper_idx).tail(full_upper_idx - full_lower_idx)
    print('Also returning "full" input and output between {0} and {1}'.format(traincopy.index[0], traincopy.index[-1]))
    
    #############################################################################################################
    # Split input and output 

    train_y = train.values[:, -1]
    train_x = train.values[:, :-1]
    test_y = test.values[:, -1]
    test_x = test.values[:, :-1]
    cv_y = cv.values[:, -1]
    cv_x = cv.values[:, :-1]

    full_y = traincopy.values[:, -1]
    full_x = traincopy.values[:, :-1]

    #############################################################################################################
    # Get time values for results

    train_idx = train.index.to_pydatetime()[:]
    print('Training set starts on {0}'.format(train_idx[1]))
    test_idx = test.index.to_pydatetime()[:]
    print('Test set starts on {0}'.format(test_idx[1]))
    cv_idx = cv.index.to_pydatetime()[:]
    print('Cross-validation set starts on {0}'.format(cv_idx[1]))

    #############################################################################################################
    # Normalise input and output

    train_x_max = train_x.max(axis=0)
    train_x_min = train_x.min(axis=0)
    train_x_ptp = train_x.ptp(axis=0)
    print('Normalising input')
    # print('Normalising input (X - min:{0}) / (max:{1} - min:{0})'.format(train_x_min, train_x_max))
    # print(train_x[1])
    train_x = (train_x - train_x_min) / train_x_ptp
    test_x = (test_x - train_x_min) / train_x_ptp
    cv_x = (cv_x - train_x_min) / train_x_ptp
    full_x = (full_x - train_x_min) / train_x_ptp

    train_y_flat = train_y.ravel() 
    train_y_max = train_y_flat.max()
    train_y_min = train_y_flat.min()
    print('Normalising input')
    # print('Normalising output (Y - min:{0}) / (max:{1} - min:{0})'.format(train_y_min, train_y_max))
    # print(train_y[1])
    train_y = (train_y - train_y_min)/(train_y_max - train_y_min)
    test_y = (test_y - train_y_min)/(train_y_max - train_y_min)
    cv_y = (cv_y - train_y_min)/(train_y_max - train_y_min)
    full_y = (full_y - train_y_min)/(train_y_max - train_y_min)

    #############################################################################################################
    # Network-type specific data prep

    nr_inputs = len(input_list) + time_inputs
    if (network_type == 'bnn') or (network_type == 'cnn') or (network_type == 'rnn_lstm') or (network_type == 'rnn_gru'):
        train_x = np.reshape(train_x, (train_x.shape[0], int(train_x.shape[1]/nr_inputs), nr_inputs))
        test_x = np.reshape(test_x, (test_x.shape[0], int(test_x.shape[1]/nr_inputs), nr_inputs))
        cv_x = np.reshape(cv_x, (cv_x.shape[0], int(cv_x.shape[1]/nr_inputs), nr_inputs))
        full_x = np.reshape(full_x, (full_x.shape[0], int(full_x.shape[1]/nr_inputs), nr_inputs))
    elif network_type == 'multi_nn':
        train_x = np.split(train_x, nr_inputs, axis=1)
        test_x = np.split(test_x, nr_inputs, axis=1)
        cv_x = np.split(cv_x, nr_inputs, axis=1)
        full_x = np.split(full_x, nr_inputs, axis=1)
    elif network_type == 'multi_cnn':
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        cv_x = np.reshape(cv_x, (cv_x.shape[0], cv_x.shape[1], 1))
        full_x = np.reshape(full_x, (full_x.shape[0], full_x.shape[1], 1))
        train_x = np.split(train_x, nr_inputs, axis=1)
        test_x = np.split(test_x, nr_inputs, axis=1)
        cv_x = np.split(cv_x, nr_inputs, axis=1)
        full_x = np.split(full_x, nr_inputs, axis=1)
    elif network_type == 'multi_cnn_custom':
        train_x = np.reshape(train_x, (train_x.shape[0], int(train_x.shape[1]/nr_inputs), nr_inputs))
        test_x = np.reshape(test_x, (test_x.shape[0], int(test_x.shape[1]/nr_inputs), nr_inputs))
        cv_x = np.reshape(cv_x, (cv_x.shape[0], int(cv_x.shape[1]/nr_inputs), nr_inputs))
        full_x = np.reshape(full_x, (full_x.shape[0], int(full_x.shape[1]/nr_inputs), nr_inputs))
    # elif network_type == 'nn_cat':
    #     print('One-hot encoding data')
    #     for cat_idx in range(1, len(cats)):
    #         train_y_cat_ids = np.where((train_y >= cats[cat_idx-1]) & (train_y < cats[cat_idx]))
    #         train_y[train_y_cat_ids] = cat_idx
    #         cv_y_cat_ids = np.where((cv_y >= cats[cat_idx-1]) & (cv_y < cats[cat_idx]))
    #         print(cv_y_cat_ids)
    #         print(cv_y.shape)
    #         cv_y[cv_y_cat_ids] = cat_idx
    #         print(cv_y.shape)
    #         test_y_cat_ids = np.where((test_y >= cats[cat_idx-1]) & (test_y < cats[cat_idx]))
    #         test_y[test_y_cat_ids] = cat_idx
    #     train_y = one_hot_encode_object_array(train_y, len(cats)-1)
    #     cv_y = one_hot_encode_object_array(cv_y, len(cats)-1)
    #     test_y = one_hot_encode_object_array(test_y, len(cats)-1)

    #############################################################################################################
    # Boom
    
    return (train_y, train_x, cv_y, cv_x, test_y, test_x, train_x_max, train_x_min, train_y_max, train_y_min, train_idx, test_idx, cv_idx, full_x, full_y)

if __name__ == '__main__':
    down_station = 'D3H008'
    up_station = 'D3H012'
    (train_y, train_x, cv_y, cv_x, test_y, test_x, train_x_max, train_x_min, train_y_max, train_y_min, train_idx, test_idx, cv_idx, full_x, full_y) = construct(down_station, [(up_station, 'linear', 0, False)], False, 96, 'nn')
    
    # np.savetxt('{0}_{1}_time_train_x.csv'.format(up_station, down_station), train_x, delimiter=',')
    # np.savetxt('{0}_{1}_time_train_y.csv'.format(up_station, down_station), train_y, delimiter=',')
    # np.savetxt('{0}_{1}_time_cv_x.csv'.format(up_station, down_station), cv_x, delimiter=',')
    # np.savetxt('{0}_{1}_time_cv_y.csv'.format(up_station, down_station), cv_y, delimiter=',')
    # np.savetxt('{0}_{1}_time_test_x.csv'.format(up_station, down_station), test_x, delimiter=',')
    # np.savetxt('{0}_{1}_time_test_y.csv'.format(up_station, down_station), test_y, delimiter=',')
    # np.savetxt('{0}_y.csv'.format(down_station), full_y, delimiter=',')

    print('train_x_min : {0}'.format(train_x_min))
    print('train_x_max : {0}'.format(train_x_max))
    print('train_y_min : {0}'.format(train_y_min))
    print('train_y_max : {0}'.format(train_y_max))
