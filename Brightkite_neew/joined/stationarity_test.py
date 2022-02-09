# will check if the probability distribution of actions [individually] are stationary or non-stationary.
import numpy as np
import pandas as pd
import csv
import os
import glob

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt

class stationarity_test:
    def __init__(self):
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
            self.process_data(user, csv_reader)
            file.close()
            break

    def adf_test(self, timeseries): #performs Augmented Dicky-Fuller test
        # ADF Test
        # Function to print out results in customised manner
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def process_data(self, user, csv_reader):
        next(csv_reader)
        data = []
        for interaction in csv_reader:
            state = interaction[1]
            state = state.strip('()')
            state = state.split('+')[0]
            print("{} {} {}".format(interaction[0], interaction[1], state))

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
