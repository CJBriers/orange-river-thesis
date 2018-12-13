""" Plot training performance """

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
       
def plot_loss(subdir):
    """Plot training performance : loss"""
    
    fig = plt.figure(facecolor='white', figsize=(20,20))
    axis = fig.add_subplot(111)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    for log_file in glob.iglob('{0}/**/log.csv'.format(subdir), recursive=True):
        print('Plotting loss for {0}'.format(log_file))
        data = np.loadtxt(log_file, delimiter=',', skiprows=1)
        plt.plot(data[:, 1], label=log_file[:-8])
    plt.ylim(0.0, 0.001)
    axes = plt.gca()
    axes.set_ylim([0,0.001])
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(prop={'size': 14})
    plt_file = '{0}/comparison_loss.png'.format(subdir)
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()
           
def plot_validation_loss(subdir):
    """Plot training performance : validation loss"""
    
    fig = plt.figure(facecolor='white', figsize=(20,20))
    axis = fig.add_subplot(111)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    for log_file in glob.iglob('{0}/**/log.csv'.format(subdir), recursive=True):
        print('Plotting validation loss for {0}'.format(log_file))
        data = np.loadtxt(log_file, delimiter=',', skiprows=1)
        plt.plot(data[:, 2], label=log_file[:-8])
    plt.ylim(0.0, 0.0005)
    axes = plt.gca()
    axes.set_ylim([0,0.0005])
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(prop={'size': 14})
    plt_file = '{0}/comparison_validation_loss.png'.format(subdir)
    plt.savefig(plt_file, bbox_inches='tight', dpi=200, papertype='a3')
    plt.close()

if __name__ == '__main__':
    folder = sys.argv[1]
    plot_loss(folder)
    plot_validation_loss(folder)
