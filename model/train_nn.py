""" ML prediction of flows at downstream station """

import os
import time
import warnings
import util
import data
import model
import predict
import keras
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") # Hide messy Numpy warnings

def train(my_model, result_dir, y_train, x_train, y_cv, x_cv, nr_epochs):
    model_file = util.model_file_name(result_dir)
    model_file_2 = util.model_file_name_lowest_cv(result_dir)
    log_file = util.log_file_name(result_dir)

    tensorboard = keras.callbacks.TensorBoard(log_dir=result_dir)
    lradapt = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.00000001, cooldown=0, min_lr=0)
    checkpoint = keras.callbacks.ModelCheckpoint(model_file, monitor='loss', save_best_only=True, mode='min', period=10)
    checkpoint_2 = keras.callbacks.ModelCheckpoint(model_file_2, monitor='val_loss', save_best_only=True, mode='min', period=1)
    early_stop_1 = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000000001, patience=20, verbose=0, mode='min')
    early_stop_2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000000000001, patience=20, verbose=0, mode='min')
    csv_log = keras.callbacks.CSVLogger(log_file, separator=',', append=True)

    print('Training model...')
    history = my_model.fit(
        x_train,
        y_train,
        validation_data=(x_cv, y_cv),
        callbacks=[checkpoint, checkpoint_2, csv_log, lradapt],
        batch_size=168,
        epochs=nr_epochs)
    print('Done training model')
    return history

def run_permutations(down_station, input_list, include_time, sample_size_list, network_type, layer_list, unit_list, nr_epochs):
    """Run permutations"""
    for i in range(0, len(layer_list)):
        for j in range(0, len(unit_list)):
            for k in range(0, len(sample_size_list)):
                run(down_station, input_list, include_time, sample_size_list[k], network_type, layer_list[i], unit_list[j], nr_epochs)

def run(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units, nr_epochs):
    """Runner"""
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # down_station, input_list, include_time, sample_size, network_type
    (y_train, x_train, y_cv, x_cv, _, _, _, _, _, _, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    
    input_dim = 0
    input_dim_2 = 0
    if (network_type == 'bnn') or (network_type == 'cnn') or (network_type == 'rnn_lstm') or (network_type == 'rnn_gru'):
        (_, input_dim, input_dim_2) = x_train.shape
    elif (network_type == 'multi_cnn'):
        input_dim = []
        for x_train_i in x_train:
            (_, input_dim_i, _) = x_train_i.shape
            input_dim.append(input_dim_i)
    else:
        (_, input_dim) = x_train.shape
    
    my_model = model.create(result_dir, input_dim, nr_layers, nr_units, network_type, input_dim_2)
    train(my_model, result_dir, y_train, x_train, y_cv, x_cv, nr_epochs)

    util.plot_training_performance(result_dir)
    predict.run(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units)

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Training time : %H:%M:%S", time.gmtime(elapsed_time_run)))

def run_prob(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units):
    """Runner"""
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, _, _, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    
    input_dim = 0
    input_dim_2 = 0
    if (network_type == 'bnn'):
        (_, input_dim, input_dim_2) = x_train.shape
    else:
        (_, input_dim) = x_train.shape

    my_model = model.create(result_dir, input_dim, nr_layers, nr_units, network_type, input_dim_2)
    trained_model_file = util.model_file_name_lowest_cv(result_dir)
    my_model.load_weights(trained_model_file, by_name=True)
    print(my_model.get_config())

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Training time : %H:%M:%S", time.gmtime(elapsed_time_run)))

def produce_diff(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units, denormalise, roundit):
    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)
    model_file = util.model_file_name(result_dir)
    my_model = util.load_model(model_file)
    data.produce_diff(my_model, result_dir, down_station, input_list, include_time, sample_size, network_type, denormalise, roundit)

if __name__ == '__main__':
    start_time = time.time()
    
    # small test run
    run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [256], 10)

    # 1 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [256], 300)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [128], 1000) # final run
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [256], 1000) # final run
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [2048], 1000) # final run
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'nn', 1, 256, True, True)

    # 2 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [2], [16, 32, 64, 128, 256, 512], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [2], [256], 250)
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'nn', 2, 256, True, True)

    # 3 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [3], [16, 32, 64, 128], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [3], [64], 332)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [3], [256], 250)

    # 4 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [4], [32, 64], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [4], [256], 250)

    # 5 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [5], [64], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [5], [256], 250)

    # cnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [128, 0], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [2, 4, 6, 8, 10], [0], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10, 8, 6, 4], [96], 100) # simple
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [8, 6, 4], [2048], 100) # with dropout
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048], 500) # final run with dropout
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'cnn', 10, 2048, True, True)

    # rnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'rnn_gru', [1], [96], 30)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'rnn_gru', [1], [96], 100)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'rnn_gru', [1], [96], 150) # final run
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'rnn_gru', 1, 96, True, True)

    # 480 time steps input

    # 1 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [480], 'nn', [1], [32, 384, 640, 768, 896, 1024, 1280], 100)
    
    # 2 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [480], 'nn', [2], [64, 128, 256], 100)

    # 3 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [480], 'nn', [3], [256], 100)

    # 4 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [480], 'nn', [4], [256], 100)

    # 5 layer fcnn
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [480], 'nn', [5], [256], 100)


    # with time
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], True, [96], 'cnn', [10], [2048], 100)
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], True, 96, 'cnn', 10, 2048, True, True)

    # with evap
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False)], False, [96], 'cnn', [11], [2048], 100)
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False)], False, 96, 'cnn', 11, 2048, True, True)

    # tributaries
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048], 500) # baseline, no tributary
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'multi_cnn', [10], [2048], 500)
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'cnn', [10], [2048], 500)
    # produce_diff('D7H012', [('D3H012', 'linear', 0, True)], False, 96, 'cnn', 10, 2048, True, True)
    # produce_diff('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, 96, 'cnn', 10, 2048, True, True)
    # produce_diff('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, 96, 'multi_cnn', 10, 2048, True, True)
    
    # with diff, no time, no evap
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'cnn', 10, 2048, False, False)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn-final-no-time', 'linear', 96, True), ('D3H008-pred-cnn-final-no-time', 'linear', 96, True)], False, [96], 'cnn', [10], [2048], 115)
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn-final-no-time', 'linear', 96, True), ('D3H008-pred-cnn-final-no-time', 'linear', 96, True)], False, 96, 'cnn', 10, 2048, True, True)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-diff-cnn-final-no-time', 'linear', 96, True)], False, [96], 'cnn', [10], [2048], 500)

    # with diff, no time, evap
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False)], False, 96, 'cnn', 10, 2048, False, False)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False), ('D3H008-real-cnn-final-evap', 'linear', 96, True), ('D3H008-pred-cnn-final-evap', 'linear', 96, True)], False, [96], 'cnn', [10], [2048], 500)

    # tributaries, with diff
    # produce_diff('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, 96, 'cnn', 10, 2048, False, False)
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('D7H012-real-cnn', 'linear', 96, True), ('D7H012-pred-cnn', 'linear', 96, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'cnn', [10], [2048], 160)

    elapsed_time = time.time() - start_time
    print(time.strftime("Total training time : %H:%M:%S", time.gmtime(elapsed_time)))
