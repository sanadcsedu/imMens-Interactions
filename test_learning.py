import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sys
import plotting
import environment2
from tqdm import tqdm
# from numba import jit, cuda
import multiprocessing
from datetime import time, datetime

class test_learning:
    def __init__(self):
        # pass
        self.time = []
        self.state = []
        self.action = []
        self.reward = []
        self.viz = []
        self.subtask = []
        self.find_states = defaultdict()
        # self.freq = defaultdict(lambda: defaultdict(float))

    def get_state(self, state):
        state = state.strip('()')
        state = state.replace(" ", "")
        state = state.split('+')

        # if state[0] not in ["Question", "Sensemaking"]:
        #     continue

        if(state[1] in ["geo-0-1", "scatterplot-0-1"]):
            new_state = "+geo_plot"
        elif state[1] in ["bar-5"]:
            new_state = "+bar_traveller"
        else:
            new_state = "+temporal"
        new_state = state[0] + new_state

        self.find_states[new_state] = 1
        return new_state

    def process_data(self, filename):
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            self.time.append(row['Time'])
            self.state.append(self.get_state(row['State']))
            self.action.append(row['action'])
            self.reward.append(row['reward'])
            self.viz.append(row['visualization'])
            self.subtask.append(row['subtask'])
            # print(row)
        # print(self.find_states)

    def functions_freq(self):
        _len = len(self.state)
        freq = defaultdict(lambda: defaultdict(float))
        sub = self.subtask[0]
        for i in range(_len):
            freq[self.state[i]][self.action[i]] += 1
            if sub != self.subtask[i]:
                print("########## {} #########".format(sub))
                for st in freq.keys():
                    sum = 0
                    for act in freq[st]:
                        sum += freq[st][act]
                    print("State: {} ______".format(st))
                    for act in freq[st]:
                        print("     {} : {:.2f} {}".format(act, freq[st][act] / sum, freq[st][act]))
                freq.clear()
                sub = self.subtask[i]

    def time_diff(self, end, start):
        etime = end.split(":")
        stime = start.split(":")
        eret = datetime(year = 2022, month = 6, day = 23, hour = int(etime[0]), minute = int(etime[1]), second = int(etime[2]))
        sret = datetime(year = 2022, month = 6, day = 23, hour = int(stime[0]), minute = int(stime[1]), second = int(stime[2]))
        delta = eret - sret
        # print(delta)
        # pdb.set_trace()
        return delta

    def function_time(self):
        _len = len(self.state)
        start = self.time[0]
        reward = 0
        for i in range(1, _len):
            reward += self.reward[i]
            # print(self.reward[i])
            if self.state[i] != self.state[i - 1] or self.action[i] != self.action[i-1]:
                end = self.time[i]
                time_delta = self.time_diff(end, start)
                # print(end)
                print("{} {} --- A: {} R:[{}] TIME: {}".format(self.subtask[i], self.state[i], self.action[i], reward, time_delta))
                reward = 0
                start = self.time[i]


if __name__ == "__main__":
    obj = test_learning()
    obj.process_data('D:\imMens Learning\QLearning\p4_reformed.csv')
    obj.functions_freq()
    obj.function_time()