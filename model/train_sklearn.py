""" Regression prediction of flows at downstream station """

import os
import time
import warnings
import util
import data
import predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

warnings.filterwarnings("ignore") # Hide messy Numpy warnings

def run_linear_regression(down_station, input_list, include_time, sample_size, network_type, nr_layers, nr_units):
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, nr_layers, nr_units, sample_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)

    #poly = PolynomialFeatures(degree=2)
    #x_train_poly = poly.fit_transform(x_train)
    #x_cv_poly = poly.fit_transform(x_cv)

    #regr = linear_model.ElasticNet(alpha=1e-3, tol=1e-9)
    #regr.fit(x_train, y_train)
    #y_pred = regr.predict(x_cv)

    # regr = linear_model.LinearRegression(fit_intercept=False)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_cv)

    util.save_linreg(regr, result_dir)
    util.save_sklearn_model(regr, result_dir)
    predict.plot_prediction(y_pred, result_dir, y_cv, train_y_max, train_y_min)

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Fitting time : %H:%M:%S", time.gmtime(elapsed_time_run)))

def run_linear_regression_permutations(down_station, input_list, include_time, sample_size_list, network_type, a, b):
    """Run permutations"""
    for k in range(0, len(sample_size_list)):
        run_linear_regression(down_station, input_list, include_time, sample_size_list[k], network_type, a, b)

def run_gpr_permutations(down_station, input_list, include_time, sample_size_list, network_type, a_list, b_list):
    """Run permutations"""
    for i in range(0, len(a_list)):
        for j in range(0, len(b_list)):
            for k in range(0, len(sample_size_list)):
                run_gpr(down_station, input_list, include_time, sample_size_list[k], network_type, include_time, a_list[i], b_list[j])

def run_gpr(down_station, input_list, include_time, sample_size, network_type, include_diff, n_estimators, b):
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, n_estimators, b, sample_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, _ ,_) = data.construct(down_station, input_list, include_time, sample_size, network_type)

    # n_estimators = 50
    gpr = BaggingRegressor(GaussianProcessRegressor(copy_X_train=False), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=1)
    # svr = SVR(C=_C, epsilon=_epsilon, verbose=True, cache_size=1024) # No bagging
    # gpr = GaussianProcessRegressor(copy_X_train=False)

    gpr.fit(x_train, y_train)
    util.save_sklearn_model(gpr, result_dir)

    y_cv_pred = gpr.predict(x_cv)
    
    predict.plot_prediction(y_cv_pred, result_dir, y_cv, train_y_max, train_y_min)

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Fitting time : %H:%M:%S", time.gmtime(elapsed_time_run)))

def run_svr_permutations(down_station, input_list, include_time, sample_size_list, network_type, c_list, epsilon_list):
    """Run permutations"""
    for i in range(0, len(c_list)):
        for j in range(0, len(epsilon_list)):
            for k in range(0, len(sample_size_list)):
                run_svr(down_station, input_list, include_time, sample_size_list[k], network_type, include_time, c_list[i], epsilon_list[j])

def run_svr(down_station, input_list, include_time, sample_size, network_type, include_diff, _C, _epsilon):
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, _C, _epsilon, sample_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)

    #svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
    #               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
    #                           "gamma": np.logspace(-2, 2, 5)})

    if network_type == 'svr':
        n_estimators = 12
        svr = BaggingRegressor(SVR(C=_C, epsilon=_epsilon, verbose=True, cache_size=768), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
        # svr = SVR(C=_C, epsilon=_epsilon, verbose=True, cache_size=1024) # No bagging
    else:
        svr = LinearSVR(C=_C, epsilon=_epsilon, verbose=1, max_iter=20000)
    svr.fit(x_train, y_train)
    util.save_sklearn_model(svr, result_dir)

    y_cv_pred = svr.predict(x_cv)
    
    predict.plot_prediction(y_cv_pred, result_dir, y_cv, train_y_max, train_y_min)

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Fitting time : %H:%M:%S", time.gmtime(elapsed_time_run)))

def produce_diff(down_station, input_list, include_time, sample_size, network_type, _C, _epsilon, denormalise, roundit):
    result_dir = util.get_result_dir(down_station, network_type, _C, _epsilon, sample_size)
    my_model = util.load_sklearn_model(result_dir)
    data.produce_diff(my_model, result_dir, down_station, input_list, include_time, sample_size, network_type, denormalise, roundit)

def run_sgdreg_permutations(down_station, input_list, include_time, sample_size_list, network_type, tol_list, eta0_list):
    """Run permutations"""
    for i in range(0, len(tol_list)):
        for j in range(0, len(eta0_list)):
            for k in range(0, len(sample_size_list)):
                run_sgdreg(down_station, input_list, include_time, sample_size_list[k], network_type, tol_list[i], eta0_list[j])

def run_sgdreg(down_station, input_list, include_time, sample_size, network_type, _tol, _eta0):
    start_time_run = time.time()

    result_dir = util.get_result_dir(down_station, network_type, _tol, _eta0, sample_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    (y_train, x_train, y_cv, x_cv, _, _, _, _, train_y_max, train_y_min, _, _, _, _, _) = data.construct(down_station, input_list, include_time, sample_size, network_type)

    sgdreg = linear_model.SGDRegressor(max_iter=100000, tol=_tol, eta0=_eta0)
    sgdreg.fit(x_train, y_train)
    y_pred = sgdreg.predict(x_cv)

    predict.plot_prediction(y_pred, result_dir, y_cv, train_y_max, train_y_min)

    elapsed_time_run = time.time() - start_time_run
    print(time.strftime("Fitting time : %H:%M:%S", time.gmtime(elapsed_time_run)))

if __name__ == '__main__':
    start_time = time.time()
    
    # thesis
    run_linear_regression_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'linreg', 1, 1)
    # produce_diff('D3H008', [('D3H012', 'linear', 0, True)], False, 96, 'linreg', 1, 1, True, True)
    # run_linear_regression_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 480], 'linreg', 1, 1)
    # run_linear_regression_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 480, 600, 720], 'linreg', 1, 1)
    # run_linear_regression_permutations('D3H008', [('D3H012', 'linear', 0, True)], False, [96], 'linreg', 1, 1)
    # run_linear_regression_permutations('D3H008', [('D3H012', 'linear', 0, True)], True, [96], 'linreg', 1, 2)

    elapsed_time = time.time() - start_time
    print(time.strftime("Total training time : %H:%M:%S", time.gmtime(elapsed_time)))
