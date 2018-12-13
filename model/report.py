""" Construct dataset """

import sys
import math
import pandas as pd
import numpy as np
import csv

def calc_gaps(station): 
    """Calculate gaps in time series"""
    df = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station), parse_dates=['Date'])
    df = df.set_index(['Date'])
    df.index = pd.to_datetime(df.index)
    dates = df.index.values

    first_date = dates[0]
    last_date = dates[-1]
    print('Data from {0} to {1}'.format(first_date, last_date))
    total_range = last_date - first_date
    total_range_seconds = total_range / np.timedelta64(1, 's')

    last_read_date = first_date
    gaps = []
    total_gap = 0;
    for d in dates:
        diff = d - last_read_date
        seconds = diff / np.timedelta64(1, 's')
        hours = diff / np.timedelta64(1, 'h')
        if hours > 72: # met stations
        # if hours > 24: # flow stations
            total_gap = total_gap + seconds
            gaps.append(seconds)
        last_read_date = d

    print('Number of gaps {0}'.format(len(gaps)))
    years = math.floor(total_gap / 3600 / 24 / 365.25)
    days = math.floor((total_gap / 3600 / 24 % 365.25))
    print('Total gap {0} years'.format(total_gap / 3600 / 24 / 365.25))
    print('Total gap {0} years {1} days'.format(years, days))

    total_left = total_range_seconds - total_gap
    years_left = math.floor(total_left / 3600 / 24 / 365.25)
    days_left = math.floor((total_left / 3600 / 24 % 365.25))
    print('Total left {0} years'.format(total_left / 3600 / 24 / 365.25))
    print('Total left {0} years {1} days'.format(years_left, days_left))

    # gap_file = '{0}-gaps.txt'.format(station)
    # np.savetxt(gap_file, gaps, delimiter=',', fmt="%s")

def calc_histogram(station):
    """Get histogram"""
    raw = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station), parse_dates=['Date'])
    raw = raw.set_index(['Date'])
    raw.index = pd.to_datetime(raw.index)
    df = raw.resample('1H').mean()
    total_count = df.count()
    i0 = df[(df['Value'] ==   0)].count()
    i1 = df[(df['Value'] >    0) & (df['Value'] <=   10)].count()
    i2 = df[(df['Value'] >   10) & (df['Value'] <=   50)].count()
    i3 = df[(df['Value'] >   50) & (df['Value'] <=  100)].count()
    i4 = df[(df['Value'] >  100) & (df['Value'] <=  200)].count()
    i5 = df[(df['Value'] >  200) & (df['Value'] <=  300)].count()
    i6 = df[(df['Value'] >  300) & (df['Value'] <=  400)].count()
    i7 = df[(df['Value'] >  400) & (df['Value'] <=  500)].count()
    i8 = df[(df['Value'] >  500) & (df['Value'] <= 1000)].count()
    i9 = df[(df['Value'] > 1000)].count()
    print('Total count: {0}'.format(total_count['Value']))
    print('         0: {0}'.format(i0['Value']/total_count['Value']))
    print('  0 -   10: {0}'.format(i1['Value']/total_count['Value']))
    print(' 10 -   50: {0}'.format(i2['Value']/total_count['Value']))
    print(' 50 -  100: {0}'.format(i3['Value']/total_count['Value']))
    print('100 -  200: {0}'.format(i4['Value']/total_count['Value']))
    print('200 -  300: {0}'.format(i5['Value']/total_count['Value']))
    print('300 -  400: {0}'.format(i6['Value']/total_count['Value']))
    print('400 -  500: {0}'.format(i7['Value']/total_count['Value']))
    print('500 - 1000: {0}'.format(i8['Value']/total_count['Value']))
    print('    > 1000: {0}'.format(i9['Value']/total_count['Value']))

def calc_histogram4(station1, station2):
    """Get histogram"""
    raw1 = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station1), parse_dates=['Date'])
    raw1 = raw1.set_index(['Date'])
    raw1.index = pd.to_datetime(raw1.index)
    df1 = raw1.resample('1H').mean()

    raw2 = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station2), parse_dates=['Date'])
    raw2 = raw2.set_index(['Date'])
    raw2.index = pd.to_datetime(raw2.index)
    df2 = raw2.resample('1H').mean()

    df1['Total'] = df1['Value'] + df2['Value']

    total_count = df1.count()

    i0 = df1[(df1['Total'] ==   0)].count()
    i1 = df1[(df1['Total'] >    0) & (df1['Total'] <=   10)].count()
    i2 = df1[(df1['Total'] >   10) & (df1['Total'] <=   50)].count()
    i3 = df1[(df1['Total'] >   50) & (df1['Total'] <=  100)].count()
    i4 = df1[(df1['Total'] >  100) & (df1['Total'] <=  200)].count()
    i5 = df1[(df1['Total'] >  200) & (df1['Total'] <=  300)].count()
    i6 = df1[(df1['Total'] >  300) & (df1['Total'] <=  400)].count()
    i7 = df1[(df1['Total'] >  400) & (df1['Total'] <=  500)].count()
    i8 = df1[(df1['Total'] >  500) & (df1['Total'] <= 1000)].count()
    i9 = df1[(df1['Total'] > 1000)].count()
    print('Total count: {0}'.format(total_count['Total']))
    print('         0: {0}'.format(i0['Total']/total_count['Total']))
    print('  0 -   10: {0}'.format(i1['Total']/total_count['Total']))
    print(' 10 -   50: {0}'.format(i2['Total']/total_count['Total']))
    print(' 50 -  100: {0}'.format(i3['Total']/total_count['Total']))
    print('100 -  200: {0}'.format(i4['Total']/total_count['Total']))
    print('200 -  300: {0}'.format(i5['Total']/total_count['Total']))
    print('300 -  400: {0}'.format(i6['Total']/total_count['Total']))
    print('400 -  500: {0}'.format(i7['Total']/total_count['Total']))
    print('500 - 1000: {0}'.format(i8['Total']/total_count['Total']))
    print('    > 1000: {0}'.format(i9['Total']/total_count['Total']))

def calc_histogram3(station1, station2, station3):
    """Get histogram"""
    raw1 = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station1), parse_dates=['Date'])
    raw1 = raw1.set_index(['Date'])
    raw1.index = pd.to_datetime(raw1.index)
    df1 = raw1.resample('1H').mean()

    raw2 = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station2), parse_dates=['Date'])
    raw2 = raw2.set_index(['Date'])
    raw2.index = pd.to_datetime(raw2.index)
    df2 = raw2.resample('1H').mean()

    raw3 = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station3), parse_dates=['Date'])
    raw3 = raw3.set_index(['Date'])
    raw3.index = pd.to_datetime(raw3.index)
    df3 = raw3.resample('1H').mean()

    df1['Total'] = df1['Value'] + df2['Value'] + df3['Value']

    total_count = df1.count()

    i0 = df1[(df1['Total'] ==   0)].count()
    i1 = df1[(df1['Total'] >    0) & (df1['Total'] <=   10)].count()
    i2 = df1[(df1['Total'] >   10) & (df1['Total'] <=   50)].count()
    i3 = df1[(df1['Total'] >   50) & (df1['Total'] <=  100)].count()
    i4 = df1[(df1['Total'] >  100) & (df1['Total'] <=  200)].count()
    i5 = df1[(df1['Total'] >  200) & (df1['Total'] <=  300)].count()
    i6 = df1[(df1['Total'] >  300) & (df1['Total'] <=  400)].count()
    i7 = df1[(df1['Total'] >  400) & (df1['Total'] <=  500)].count()
    i8 = df1[(df1['Total'] >  500) & (df1['Total'] <= 1000)].count()
    i9 = df1[(df1['Total'] > 1000)].count()
    print('Total count: {0}'.format(total_count['Total']))
    print('         0: {0}'.format(i0['Total']/total_count['Total']))
    print('  0 -   10: {0}'.format(i1['Total']/total_count['Total']))
    print(' 10 -   50: {0}'.format(i2['Total']/total_count['Total']))
    print(' 50 -  100: {0}'.format(i3['Total']/total_count['Total']))
    print('100 -  200: {0}'.format(i4['Total']/total_count['Total']))
    print('200 -  300: {0}'.format(i5['Total']/total_count['Total']))
    print('300 -  400: {0}'.format(i6['Total']/total_count['Total']))
    print('400 -  500: {0}'.format(i7['Total']/total_count['Total']))
    print('500 - 1000: {0}'.format(i8['Total']/total_count['Total']))
    print('    > 1000: {0}'.format(i9['Total']/total_count['Total']))

def calc_histogram2(station):
    """Get histogram"""
    raw = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station), parse_dates=['Date'])
    raw = raw.set_index(['Date'])
    raw.index = pd.to_datetime(raw.index)
    df = raw.resample('1H').mean()
    total_count = df.count()
    i0 = df[(df['Value'] ==   0)].count()
    i1 = df[(df['Value'] >    0) & (df['Value'] <=    5)].count()
    i2 = df[(df['Value'] >    5) & (df['Value'] <=   10)].count()
    i3 = df[(df['Value'] >   10) & (df['Value'] <=   20)].count()
    i4 = df[(df['Value'] >   20) & (df['Value'] <=   50)].count()
    i5 = df[(df['Value'] >   50) & (df['Value'] <=  100)].count()
    i6 = df[(df['Value'] >  100)].count()
    print('Total count: {0}'.format(total_count['Value']))
    print('        0: {0}'.format(i0['Value']/total_count['Value']))
    print('  0 -   5: {0}'.format(i1['Value']/total_count['Value']))
    print('  5 -  10: {0}'.format(i2['Value']/total_count['Value']))
    print(' 10 -  20: {0}'.format(i3['Value']/total_count['Value']))
    print(' 20 -  50: {0}'.format(i4['Value']/total_count['Value']))
    print(' 50 - 100: {0}'.format(i5['Value']/total_count['Value']))
    print('    > 100: {0}'.format(i6['Value']/total_count['Value']))

def median_sampling_rate(station):
    """Get median over year sampling rate"""
    raw = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station), parse_dates=['Date'])
    raw = raw.set_index(['Date'])
    raw.index = pd.to_datetime(raw.index)
    df = raw.resample('Y').count()
    df.to_csv('{0}_sample_count.csv'.format(station))

def resample(station):
    """Resample station data"""
    raw = pd.read_csv('../data/all_clean/{0}-clean.txt'.format(station), parse_dates=['Date'])
    raw = raw.set_index(['Date'])
    raw.index = pd.to_datetime(raw.index)
    df = raw.resample('1H').mean()
    df = df.round({'Value': 0})
    df.to_csv('{0}_resampled.csv'.format(station))

if __name__ == '__main__':
    station = sys.argv[1]
    calc_gaps(station)
    #calc_histogram(station)
    #calc_histogram2(station)
    #calc_histogram3('D7H014Z', 'D7H015Z', 'D7H016Z')
    #calc_histogram4('D7H008', 'D7H017PLUS')
    #median_sampling_rate(station)
    #resample(station)
    