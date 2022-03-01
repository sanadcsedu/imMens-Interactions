# will check if the probability distribution of actions [individually] are stationary or non-stationary.
import pdb
import random

import numpy as np
import pandas as pd
import csv
import os
import glob

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pymannkendall as mk
import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict

import matplotlib.pyplot as plt

class stationarity_test:
    def __init__(self):
        self.key_list = []
        pass

    def get_files(self):
        #read raw interaction file names
        self.file_names = glob.glob("*_reformed.csv")
        # files = sorted(os.listdir('.'))
        for names in self.file_names:
            file = open(names, 'r')
            csv_reader = csv.reader(file)
            # pdb.set_trace()
            user = names.split('-')[0]
            print(user)
            # new_fname = user + '_raw.csv'
            # state = 'Question+geo-0-1'
            self.process_data(user, csv_reader)
            file.close()
            self.key_list = []
            # break

    def kpss_test(self, timeseries):
        print('Results of KPSS Test:')
        kpsstest = kpss(timeseries, regression='ct')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        for key, value in kpsstest[3].items():
            kpss_output['Critical Value (%s)' % key] = value
        # print(kpss_output)
        # Checking whether the null hypothesis should be rejected
        # For KPSS Null Hypothesis: The series is trend stationary or has no unit root
        if kpsstest[1] < 0.05:
            print("KPSS Non-Stationary")
        else:
            print("KPSS Stationary")

    def adf_test(self, timeseries): #performs Augmented Dicky-Fuller test
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        # print(dfoutput)
        #Checking whether the null hypothesis should be rejected
        #For ADF Null Hypothesis: The series is non-stationary or has a unit root
        if dftest[1] < 0.05: #Reject
            print("ADF Stationary")
        else:
            print("ADF Non-Stationary")

    def mk_test(self, timeseries): #performs Mann Kendall Trend test
        print('Results of Mann Kendall Trend Test:')
        mktest = mk.original_test(timeseries)
        # print(mktest)
        print("Trend: {}".format(mktest[0]))
        print("h: {}".format(mktest[1]))
        print("p-value: {:.3f}".format(mktest[2]))

    def get_key(self):
        ch = random.randint(1, 100)
        z = 0
        while True:
            if ch not in self.key_list:
                break
            ch = random.randint(1, 100)
            z += 1
            # pdb.set_trace()
        self.key_list.append(ch)
        return ch

    #Curates the dataset based on state
    def curate_data(self, state, data):
        sz = len(data)
        curated_data = []
        _map = defaultdict(float)
        test_data = []
        for i in range(sz):
            if data[i][1] == state:
                _map[data[i][2]] += 1
                #Here I'm checking the probability of action 'Same'
                probs = (_map['same'])/(_map['same'] + _map['change'])
                curated_data.append((data[i][0], data[i][1], data[i][2], probs))
                test_data.append(probs)
        # for items in curated_data:
        #     print(items)
        # if len(test_data) <= 10:
        #     return
        self.mk_test(test_data)

    def process_data(self, user, csv_reader):
        next(csv_reader)
        data = []
        prev_state = None
        flag = False
        _map = defaultdict(int)
        # key = 1
        for interaction in csv_reader:
            cur_state = interaction[1]
            if prev_state == cur_state:
                action = "same"
            else:
                action = "change"
            # print("{} {} {}".format(state, probs, action1))
            # if _map[(prev_state, action)] == 0:
            #     _map[(prev_state, action)] = self.get_key()
            #
            if flag:
                data.append((interaction[0], prev_state, action))
            flag = True
            prev_state = cur_state
            _map[prev_state] = 1
        self.curate_data('(Sensemaking+bar-5)', data)
        # print("States: ")
        # for z in _map.keys():
        #     print(z)

        # startpoint = int(dataset_size * 0.15)
        # t = 0
        # for st, act in data:
        #     print("{} {} {}".format(t, st, act))
        #     t += 1
        # for items in _map:
        #     print("{} {}".format(items, _map[items]))
        # self.adf_test(data[startpoint:])
        # print("\n")
        # self.kpss_test(data[startpoint:])
        # print("\n")
        # self.mk_test(data)

    def data(self):
        # Load the dataset
        df = sm.datasets.sunspots.load_pandas().data
        # Check the dimensionality of the dataset
        df.shape
        print("Dataset has {} records and {} columns".format(df.shape[0], df.shape[1]))
        # Changing the YEAR data type and setting it as index
        df['YEAR'] = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
        df.index = df['YEAR']
        # Check the data type
        del df['YEAR']
        # View the dataset
        df.head()
        # Plotting the Data
        # Define the plot size
        plt.figure(figsize=(16, 5))
        # Plot the data
        plt.plot(df.index, df['SUNACTIVITY'], label="SUNACTIVITY")
        plt.legend(loc='best')
        plt.title("Sunspot Data from year 1700 to 2008")
        # plt.show()
        print(df['SUNACTIVITY'].shape)
        self.adf_test(df['SUNACTIVITY'])

if __name__ == "__main__":
    obj = stationarity_test()
    obj.get_files()
