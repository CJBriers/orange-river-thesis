""" ML prediction of flows at downstream station """

import os
import time
import warnings
import util
import data
import model
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") # Hide messy Numpy warnings

def predict(my_model, result_dir, x, y, train_y_max, train_y_min):
    start_time = time.time()
    
    prediction_temp = my_model.predict(x)
    prediction = np.reshape(prediction_temp, (prediction_temp.size,))

    elapsed_time = time.time() - start_time
    print(time.strftime("Prediction done : %H:%M:%S", time.gmtime(elapsed_time)))
    print("{0} predictions in {1} seconds ({2} per second)".format(len(prediction), elapsed_time, len(prediction)/elapsed_time))

    plot_prediction(prediction, result_dir, y, train_y_max, train_y_min) # uncomment for normal prediction
    # plot_prediction_compare_with_mike11(prediction, result_dir, y, train_y_max, train_y_min) # uncomment for DWS prediction

def plot_prediction_compare_with_mike11(prediction, result_dir, y, train_y_max, train_y_min):
    # denormalise data
    prediction_denormalised = prediction * (train_y_max - train_y_min) + train_y_min
    y_denormalised = y * (train_y_max - train_y_min) + train_y_min

    rmse = util.rmse(prediction_denormalised, y_denormalised)
    f = open('{0}/rmse_compare_{1}'.format(result_dir, rmse), "w+")
    f.close()

    mape = util.mean_absolute_percentage_error(prediction_denormalised, y_denormalised)
    f = open('{0}/mape_compare_{1}'.format(result_dir, mape), "w+")
    f.close()

    r2 = util.r2(prediction_denormalised, y_denormalised)
    f = open('{0}/r2_compare_{1}'.format(result_dir, r2), "w+")
    f.close()

    plt_file = '{0}/plot_compare_{1}.png'.format(result_dir, mape)
    util.plot_results_unordered_compare_with_mike11(prediction_denormalised, y_denormalised, plt_file)

    mape_batch = util.mean_absolute_percentage_error(prediction_denormalised, y_denormalised)
    plt_file = '{0}/plot_compare_{1}.png'.format(result_dir, mape_batch)
    util.plot_results_unordered(prediction_denormalised, y_denormalised, plt_file)


def plot_prediction(prediction, result_dir, y, train_y_max, train_y_min):
    # denormalise data
    prediction_denormalised = prediction * (train_y_max - train_y_min) + train_y_min
    y_denormalised = y * (train_y_max - train_y_min) + train_y_min

    rmse = util.rmse(prediction_denormalised, y_denormalised)
    f = open('{0}/rmse_{1}'.format(result_dir, rmse), "w+")
    f.close()

    mape = util.mean_absolute_percentage_error(prediction_denormalised, y_denormalised)
    f = open('{0}/mape_{1}'.format(result_dir, mape), "w+")
    f.close()

    r2 = util.r2(prediction_denormalised, y_denormalised)
    f = open('{0}/r2_{1}'.format(result_dir, r2), "w+")
    f.close()

    plt_file = '{0}/plot_unordered_{1}.png'.format(result_dir, mape)
    util.plot_results_unordered(prediction_denormalised, y_denormalised, plt_file)

    # output in batches of 1000 hours (+- 40 days)
    step_size = 1000
    for x in range(0, prediction_denormalised.size-step_size, step_size):
        mape_batch = util.mean_absolute_percentage_error(prediction_denormalised[x:x+step_size], y_denormalised[x:x+step_size])
        plt_file = '{0}/plot_unordered_{1}_{2}.png'.format(result_dir, mape_batch, x)
        util.plot_results_unordered(prediction_denormalised[x:x+step_size], y_denormalised[x:x+step_size], plt_file)

    # plt_file = '{0}/plot_confusion_matrix.png'.format(result_dir)
    # util.plot_confusion_matrix(prediction_denormalised, y_denormalised, plt_file)

def run_permutations(down_station, input_list, include_time, sample_size_list, network_type, layer_list, unit_list):
    """Run permutations"""
    for i in range(0, len(layer_list)):
        for j in range(0, len(unit_list)):
            for k in range(0, len(sample_size_list)):
                run(down_station, input_list, include_time, sample_size_list[k], network_type, layer_list[i], unit_list[j])

def run(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units):
    """Runner"""
    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)

    util.plot_training_performance(result_dir)
    
    model_file = util.model_file_name(result_dir)
    # model_file = util.model_file_name_lowest_cv(result_dir) # lowest cv model
    my_model = util.load_model(model_file)

    # uncomment for DWS prediction
    # for specific dates, see internals of data.construct
    #(_, _, _, _, _, _, _, _, train_y_max, train_y_min, _, _, _, full_x, full_y) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    #predict(my_model, result_dir, full_x, full_y, train_y_max, train_y_min)
    
    # uncomment for normal prediction
    #(_, _, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, full_x, full_y) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    #predict(my_model, result_dir, x_cv, y_cv, train_y_max, train_y_min)

    # uncomment for test prediction
    (_, _, _, _, y_test, x_test, _, _, train_y_max, train_y_min, _, _, _, full_x, full_y) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    predict(my_model, result_dir, x_test, y_test, train_y_max, train_y_min)

    # uncomment for full prediction
    # (_, _, _, _, _, _, _, _, train_y_max, train_y_min, _, _, _, full_x, full_y) = data.construct(down_station, input_list, include_time, sample_size, network_type)
    # predict(my_model, result_dir, full_x, full_y, train_y_max, train_y_min)
    
def run_prob(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units):
    """Runner"""
    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)

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

    predict(my_model, result_dir, x_cv, y_cv, train_y_max, train_y_min)

def run_sklearn(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units):
    """Runner"""
    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)

    (_, _, y_cv, x_cv, y_test, x_test, _, _, train_y_max, train_y_min, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)

    my_model = util.load_sklearn_model(result_dir)

    start_time = time.time()    
    # y_pred = my_model.predict(x_cv) # uncomment for validation
    y_pred = my_model.predict(x_test) # uncomment for test
    elapsed_time = time.time() - start_time
    print(time.strftime("Prediction done : %H:%M:%S", time.gmtime(elapsed_time)))
    print("{0} predictions in {1} seconds ({2} per second)".format(len(y_pred), elapsed_time, len(y_pred)/elapsed_time))

    # plot_prediction(y_pred, result_dir, y_cv, train_y_max, train_y_min) # uncomment for validation
    plot_prediction(y_pred, result_dir, y_test, train_y_max, train_y_min) # uncomment for test

if __name__ == '__main__':
    # run('C9R003', ['C5H014', 'C9H024'], [], 1, 1, 192, 1000.0, 'rnn_lstm', [])
    # run('D3H008', ['D3H012'], [], 1, 96, 192, 'rnn_gru', [])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn', 'linear', 96, True), ('D3H008-pred-cnn', 'linear', 96, True)], False, [96], 'multi_cnn', [10], [2048])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn', 'linear', 96, True), ('D3H008-pred-cnn', 'linear', 96, True)], False, [96], 'cnn', [10], [2048])
    # run_permutations('D3H008', ['D3H012'], [], [10], [2048], [96], 'multi_cnn', [], False, True)
    # run('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn', 'linear', 96, True), ('D3H008-pred-cnn', 'linear', 96, True)], False, 96, 'cnn', 10, 2048)
    # run('D8H003', ['D3H012', 'C9R003'], [], 10, 2048, 384, 'multi_cnn', [], True)
    
    # thesis
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [32, 64, 128, 256, 512, 1024, 2048, 4096])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [2], [16, 32, 64, 128, 256, 512])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [3], [16, 32, 64, 128])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [4], [32, 64])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [5], [64])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False)], False, [96], 'cnn', [10], [2048])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'rnn_gru', [1], [96])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn-final-no-time', 'linear', 96, True), ('D3H008-pred-cnn-final-no-time', 'linear', 96, True)], False, [96], 'cnn', [10], [2048])
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('D7H012-real-cnn', 'linear', 96, True), ('D7H012-pred-cnn', 'linear', 96, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'cnn', [10], [2048])

    # TEST
    run_sklearn('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'linreg', 1, 1)
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [1], [256])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'nn', [2], [256])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048])
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'rnn_gru', [1], [96])

    # seasonal
    # no-time
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'cnn', [10], [2048])
    # time
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True)], True, [96], 'cnn', [11], [2048])
    # evap
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3E003', 'linear', 96, False)], False, [96], 'cnn', [12], [2048])
    # diff
    # run_permutations('D3H008', [('D3H012', 'linear', 0, True), ('D3H008-real-cnn-final-no-time', 'linear', 96, True), ('D3H008-pred-cnn-final-no-time', 'linear', 96, True)], False, [96], 'cnn', [13], [2048])

    # tributaries
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'multi_cnn', [10], [2048])
    # run_permutations('D7H012', [('D3H012', 'linear', 0, True), ('C5H039', 'linear', 0, False), ('C5H049', 'linear', 0, False), ('C9H021', 'linear', 0, False)], False, [96], 'cnn', [10], [2048])
    