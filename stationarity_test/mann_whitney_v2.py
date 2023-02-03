#This version of the Mann Whitney test checks stationarity of the actions based on frequency of picked visualizations
import csv
import pdb
import glob
import random

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.stats import mannwhitneyu


class integrate:
    def __init__(self):
        self.raw_files = glob.glob("D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\RawInteractions\\*.csv")
        self.excel_files = glob.glob("D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\FeedbackLog\\*.xlsx")
        self.path = 'D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\Merged\\'
        self.vizs = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
        self.freq = defaultdict(list)

    def debug(self, user, raw_data, feedback_data):
        print(user)
        print("#######RAW#########")
        for idx in range(len(raw_data)):
            print(idx, raw_data[idx][0], raw_data[idx][1], raw_data[idx][2])
        print("#######EXCEL#########")
        for idx in range(len(feedback_data)):
            print(idx, feedback_data[idx][0], feedback_data[idx][2], feedback_data[idx][3])

    def get_files(self):
        for_now = ['p7', 'p2', 'p11', 'p3']
        for raw_fname in self.raw_files:
            user = Path(raw_fname).stem.split('-')[0]
            if user not in for_now:
                continue
            excel_fname = [string for string in self.excel_files if user in string][0]
            self.freq.clear()
            self.merge(user, raw_fname, excel_fname)

    def excel_to_memory(self, df):
        data = []
        subtask_end_time = defaultdict(int)
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            if row['State'] == "None": #When reading from excel do not consider states = None
                continue
            # data.append([seconds, row['proposition'], row['Reward'], row['State'], row['Subtask']])
            subtask_end_time[row['Subtask']] = seconds
        return subtask_end_time

    def raw_to_memory(self, csv_reader, subtask_end_time):
        next(csv_reader)
        data = []
        for lines in csv_reader:
            time = lines[0].split(":")
            mm = int(time[1])
            ss = int(time[2])
            seconds = mm * 60 + ss
            get_subtask = self.get_cur_subtask(seconds, subtask_end_time)
            if get_subtask is not None:
                data.append([seconds, lines[1], lines[2], get_subtask])
            else:
                break
        return data

    def get_cur_subtask(self, cur_time, subtask_end_time):
        for keys in subtask_end_time:
            if cur_time <= subtask_end_time[keys]:
                return keys
        return None

    #Just using the raw interactions to check the frequency of picked arms (visualization)
    #Also tracking which part of the interactions belongs to which tasks.
    def merge(self, user, raw_fname, excel_fname):
        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet3 (2)", usecols="A:G")
        subtask_end_time = self.excel_to_memory(df_excel)

        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader, subtask_end_time)
        raw_interaction.close()

        for idx in range(len(raw_data)):
            for v in self.vizs:
                if v == raw_data[idx][2]:
                    self.freq[v].append(1)
                else:
                    self.freq[v].append(0)

        # self.stationarity_test1(user)

        windows = []
        self.freq.clear()
        cur_subtask = 1
        idx = 0
        while idx < len(raw_data):
            if raw_data[idx][3] == cur_subtask:
                for v in self.vizs:
                    if v == raw_data[idx][2]:
                        self.freq[v].append(1)  # Adding 1 if we see the user picking this viz
                    else:
                        self.freq[v].append(0)  # Adding 0 as the user is not picking this viz
                idx += 1
            else:
                windows.append(self.freq.copy())
                self.freq.clear()
                cur_subtask = raw_data[idx][3]
                # idx -= 1
            # pdb.set_trace()
        windows.append(self.freq.copy())
        # pdb.set_trace()
        self.stationarity_test2(user, windows)

    # This one checks non-stationarity for 50-50 split
    def stationarity_test1(self, user):
        # print("##### USER {} #######".format(user))
        for v in self.vizs:
            w1 = []
            w2 = []
            sz = len(self.freq[v])
            # pdb.set_trace()
            numer = 0
            #In this implementation, the probablity of the 2nd half is not starting from 0
            #As a result, when we do the mann-whitney test, the 2nd probability series considers what happend in the first half. (Past history)
            for idx in range(len(self.freq[v])):
                denom = idx + 1
                numer += self.freq[v][idx]
                if idx * 2 < sz:
                    w1.append(round(numer / denom, 2))
                else:
                    w2.append(round(numer / denom, 2))
            print(v, end=" ")
            try:
                results = mannwhitneyu(w1, w2)
                # print(results)
                if results.pvalue < 0.05:
                    print("True", end=" ")
                else:
                    print("False", end=" ")
            except ValueError as ve:
                print("NED", end=" ")
            print()

    #This one does stationarity checks for all windows (subtasks)
    def stationarity_test2(self, user, windows):
        print("##### USER {} #######".format(user))
        # Generating the probability distribution from the windows
        S = NS = NED = 0
        for v in self.vizs:
            windows_prob = []
            for w in windows:
                w1 = []
                numer = 0
                # Also in this implementation we are not starting the subtask probability from 0
                for idx in range(len(w[v])):
                    numer += w[v][idx]
                    w1.append(round(numer / (idx + 1), 2))
                windows_prob.append(w1)
            # Calculate the number of time we see stationarity, non-stationarity and NED
            # S = NS = NED = 0
            # print(v)
            # print("-------------")
            for idx in range(len(windows_prob)):
                # print(idx+1, end= "------\n")
                for idx2 in range(idx+1, len(windows_prob)):
                    # print("{}, {}, {}, {}, ".format(user, v, idx+1, idx2 + 1), end=" ")
                    try:
                        results = mannwhitneyu(windows_prob[idx], windows_prob[idx2])
                        if results.pvalue < 0.05: #Non-stationary
                            # print("True", end=" ")
                            NS += 1
                        else: #Stationary
                            # print("False", end=" ")
                            S += 1
                    except ValueError as ve: #Not Enough data
                        # print("NED", end=" ")
                        NED += 1
                    # print()
        print("Non-Stationarity {}".format(round(NS / (S + NS + NED), 2)))
        print("Stationarity {}".format(round(S / (S + NS + NED), 2)))
        print("NED {}".format(round(NED / (NED + NS + NED), 2)))

    def plot_graph(self, user):
        for v in self.cum_rewards:
            x_axis = []
            y_axis = []
            for keys in self.cum_rewards[v]:
                x_axis.append(keys[0])
                y_axis.append(keys[1])

            plt.plot(x_axis, y_axis, label=v)
            plt.ylabel('Rewards')
            # plt.xticks([])
            plt.xlabel('time')
        plt.legend(loc='best')
        title = 'Cumulative Rewards for user: ' + user
        plt.title(title)
        plt.show()

    # def bootstrapping(self, user):

if __name__ == "__main__":
    obj = integrate()
    obj.get_files()