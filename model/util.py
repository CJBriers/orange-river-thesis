""" Util methods """

import os
import itertools
import keras
import h5py
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import r2_score

def sklearn_model_file_name(result_dir):
    """Model file name for trained sklearn model"""
    return '{0}/model.pkl'.format(result_dir)

def sklearn_model_2_file_name(result_dir):
    """Model file name for trained sklearn model"""
    return '{0}/model_2.pkl'.format(result_dir)

def model_file_name(result_dir):
    """Model file name for lowest loss"""
    return '{0}/model.h5'.format(result_dir)

def model_file_name_lowest_cv(result_dir):
    """Model file name for lowest validation loss"""
    return '{0}/model_2.h5'.format(result_dir)

def log_file_name(result_dir):
    """Model training performance log"""
    return '{0}/log.csv'.format(result_dir)

def linreg_file_name(result_dir):
    """Linreg coefficients"""
    return '{0}/linreg_coef.csv'.format(result_dir)

def plot_training_performance(result_dir):
    """Plot training performance"""
    log_file = log_file_name(result_dir)
    data = np.loadtxt(log_file, delimiter=',', skiprows=1)

    fig = plt.figure(facecolor='white', figsize=(20,20))
    axis = fig.add_subplot(111)
    plt.plot(data[:, 1], label='Loss')
    plt.plot(data[:, 2], label='Validation Loss')
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(prop={'size': 14})
    plt_file = result_dir + "/train_performance.png"
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()

def plot_results_unordered(predicted_data, true_data, plt_file):
    """Plot actual vs predicted results"""
    fig = plt.figure(facecolor='white', figsize=(20,5))
    # fig = plt.figure(facecolor='white', figsize=(20,15)) # uncomment for DWS plot
    axis = fig.add_subplot(111)
    
    axis.plot(true_data, label='Truth') # comment for non-DWS plot
    plt.plot(predicted_data, label='Predicted') # comment for non-DWS plot
    
    # plt.plot(predicted_data, label='Modelled flow', color='black', linewidth=1) # uncomment for DWS plot
    # plt.plot(true_data, label='Recorded flow', color='green', linewidth=0.75) # uncomment for DWS plot

    plt.ylabel('$Flow (m^3/s)$', fontsize=14)
    plt.xlabel('Time (hours)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    # plt.ylim(0, 150) # uncomment for DWS plot
    plt.legend(prop={'size': 14})
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()

def plot_results_unordered_compare_with_mike11(predicted_data, true_data, plt_file):
    """Plot actual vs predicted results"""
    fig = plt.figure(facecolor='white', figsize=(20,15))
    axis = fig.add_subplot(111)
    
    plt.plot(predicted_data, color='black', linewidth=1)
    plt.plot(true_data, color='green', linewidth=0.75)

    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(0, 150)
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()

def plot_results_ordered(predicted_data, true_data, plt_file):
    """Plot actual vs predicted results"""
    true_data_sorted, predicted_data_sorted = zip(*sorted(zip(true_data, predicted_data)))
    fig = plt.figure(facecolor='white')
    fig.add_subplot(111)
    plt.scatter(true_data_sorted, predicted_data_sorted, label='Truth vs Predicted')
    plt.legend()
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()

def plot_confusion_matrix(predicted_data, true_data, plt_file):
    # true_buckets = np.floor(np.sqrt(true_data + 4.0))
    # predicted_buckets = np.floor(np.sqrt(predicted_data + 4.0))
    true_buckets = np.copy(true_data)
    predicted_buckets = np.copy(predicted_data)

    buckets = [
        0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
        110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 
        210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 
        320, 340, 360, 380, 400, 
        420, 440, 460, 480, 500, 
        550, 600, 
        650, 700, 
        750, 800, 
        850, 900, 
        950, 1000, 
        10000
    ]
    for i in range(0, len(buckets) - 1):
        true_mask = np.logical_and(true_data >= buckets[i], true_data < buckets[i+1])
        true_buckets[true_mask] = i
        predicted_mask = np.logical_and(predicted_data >= buckets[i], predicted_data < buckets[i+1])
        predicted_buckets[predicted_mask] = i

    cm = metrics.confusion_matrix(true_buckets, predicted_buckets)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)

    fig = plt.figure(facecolor='white')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    plt.xticks(np.arange(18), (
        'Low\nflow', '', '', '', '', 
        '', '', '', '', '', 
        '', '', '', 
        '', '', '', '', 'High\nflow'
    ))
    plt.yticks(np.arange(18), (
        'Low\nflow', '', '', '', '', 
        '', '', '', '', '', 
        '', '', '', 
        '', '', '', '', 'High\nflow'
    ))
    plt.tick_params(axis='both', labelsize=10, length=0)

    plt.xlabel('xlabel', fontsize=12)
    plt.ylabel('ylabel', fontsize=12)

    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.05:
            txt = format(cm[i, j], fmt)    
        else:
            txt = ''
        
        plt.text(j, i, txt,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=4.0)

    plt.tight_layout()
    plt.ylabel('Truth')
    plt.xlabel('Predicted')
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')

def rmse(predictions, targets):
    """Calculate root mean square error"""
    return np.sqrt(((predictions - targets) ** 2).mean())

def r2(predictions, targets):
    """Calculate coefficient of determination"""
    return r2_score(targets, predictions)

def mean_absolute_percentage_error(predictions, targets):
    """Calculate mean absolute percentage error"""
    mask = (targets != 0.0)
    return (np.fabs(targets - predictions)/targets)[mask].mean()*100.0

def get_result_dir(down_station, model_type, nr_layers, nr_units, sample_size):
    return './results/{0}/{1}_{2}_{3}_{4}'.format(down_station, model_type, nr_layers, nr_units, sample_size)

def get_old_result_dir(down_station, model_type, nr_layers, nr_units, sample_size):
    return './results/{0}/{1}/{2}/{3}/{4}'.format(down_station, model_type, nr_layers, nr_units, sample_size)

def load_model(model_file):
    """Load best trained weights"""
    if os.path.isfile(model_file):
        print('Loaded model from {0}'.format(model_file))
        return keras.models.load_model(model_file)
    else:
        print('No model loaded from {0}'.format(model_file))
        return None

def save_linreg(linreg, result_dir):
    """Save sklearn linreg model coefficients"""
    np.savetxt(linreg_file_name(result_dir), linreg.coef_, delimiter=",")
    print('sklearn model saved')

def save_sklearn_model(svr, result_dir):
    """Save sklearn model"""
    joblib.dump(svr, sklearn_model_file_name(result_dir))
    print('sklearn model saved')

def load_sklearn_model(result_dir):
    """Get sklearn model"""
    return joblib.load(sklearn_model_file_name(result_dir))

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()
