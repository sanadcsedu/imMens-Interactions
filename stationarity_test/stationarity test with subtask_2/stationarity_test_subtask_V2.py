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
            print("$$$$$$$$$  " + user + "  $$$$$$$$$")
            # new_fname = user + '_raw.csv'
            # state = 'Question+geo-0-1'
            self.process_data_v2(user, csv_reader)
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
        # print("h: {}".format(mktest[1]))
        # print("p-value: {:.3f}".format(mktest[2]))

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

    #Curates the dataset based on state,action pair
    #if (sensemaking,same) (Question,change) -> +1
    #else if (Question, same) (Sensemaking, Change) -> -1

    def curate_data(self, data, label):
        # print("#############################")
        print("Currently working with " + label)
        # print("#############################")

        sz = len(data)
        test_data = []
        for i in range(sz):
            if data[i][1] == 'Sensemaking' and data[i][2] == 'same':
                test_data.append(1)
            elif data[i][1] == 'Question' and data[i][2] == 'change':
                test_data.append(1)
            elif data[i][1] == 'Question' and data[i][2] == 'same':
                test_data.append(-1)
            else: # 'Sensemaking' and 'change':
                test_data.append(-1)

        self.mk_test(test_data)

    # Curates the dataset based on state only
    # if State = Sensemaking -> +1
    # else State = Question -> -1

    def curate_data_v2(self, data, label):
        # print("#############################")
        print("Currently working with " + label)
        # print("#############################")
        sz = len(data)
        test_data = []
        for i in range(sz):
            if data[i][1] == 'Sensemaking': # State = 'Sensemaking':
                test_data.append(1)
            else:  # State = 'Question':
                test_data.append(-1)

        self.mk_test(test_data)

    def get_state(self, state):
        state = state.strip('()')
        state = state.split('+')[0]
        return state

    def process_data(self, user, csv_reader):
        next(csv_reader)

        data_2d = []

        prev_state = None
        flag = False
        _map = defaultdict(int)
        # key = 1
        data_1d = []
        subtask = '1'
        for interaction in csv_reader:
            cur_state = self.get_state(interaction[1])
            if cur_state not in ('Question', 'Sensemaking'):
                continue
            if prev_state == cur_state:
                action = "same"
            else:
                action = "change"

            if flag:
                if interaction[5] == subtask:
                    data_1d.append((interaction[0], prev_state, action))
                else:
                    data_2d.append(data_1d)
                    data_1d = []
                    subtask = interaction[5]
                    data_1d.append((interaction[0], prev_state, action))
                # print("{} {} {}".format(interaction[0], prev_state, action))

            flag = True
            # pdb.set_trace()

            prev_state = cur_state
            _map[prev_state] += 1

        # pdb.set_trace()
        if len(data_1d) > 1:
            data_2d.append(data_1d)

        for i in range(len(data_2d)):
            _label = "Subtask " + str(i + 1)
            self.curate_data_v2(data_2d[i], _label)

    def process_data_v2(self, user, csv_reader):
        next(csv_reader)

        data_2d = []

        prev_state = None
        flag = False
        _map = defaultdict(int)
        # key = 1
        data_1d = []
        subtask = '1'
        for interaction in csv_reader:
            cur_state = self.get_state(interaction[1])
            if cur_state not in ('Question', 'Sensemaking'):
                continue
            if prev_state == cur_state:
                action = "same"
            else:
                action = "change"

            if flag:
                data_1d.append((interaction[0], prev_state, action))

            flag = True
            # pdb.set_trace()

            prev_state = cur_state
            _map[prev_state] += 1

        self.curate_data(data_1d, "Full Task")


if __name__ == "__main__":
    obj = stationarity_test()
    obj.get_files()
