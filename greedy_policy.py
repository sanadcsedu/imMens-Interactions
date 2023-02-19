import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import misc
import random
import sys
sys.path.insert(0, '/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/')
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import adaptive_epsilon
from mortal_bandit import mortal_bandit


class Greedy:
    def __init__(self, vizs):
        self.vizs = vizs
        self.arms_idx = defaultdict()
        idx = 0
        for v in self.vizs:
            self.arms_idx[v] = idx
            idx += 1
        self.arms = np.full(4, 1, dtype='float')
        # print(self.arms)

    def run(self, data, threshold, version):
        epoch = 10
        accu_list = []
        for thres in threshold: #Running for all thresholds

            #Training Module
            sz = len(data)
            s_idx = int(thres * sz) # Used for partitioning the dataset

            for idx in range(0, s_idx):
                self.arms[self.arms_idx[data[idx][1]]] += data[idx][2]
            sum = self.arms.sum()
            self.arms /= sum
            # pdb.set_trace()
            arg_max = np.argmax(self.arms) #Finds the arm that yielded the maximum reward
            arg_value = self.arms[arg_max] # Probability
            accu = 0
            if version == "V1":
                # Testing module V1: In this code-snippet the Greedy policy is to repeatedly pick the arm with maximum reward
                for e in range(epoch):
                    cnt = 0
                    ###### Change #######
                    sum = self.arms.sum()
                    self.arms /= sum
                    ####################
                    cur_arm = self.vizs[arg_max] #Always picking the best action based on past experience.
                    denom = 0
                    for idx in range(s_idx, sz):
                        denom += 1
                        if data[idx][1] == cur_arm:
                            cnt += 1
                            self.arms[self.arms_idx[data[idx][1]]] += data[idx][2]
                        else:
                            self.arms[self.arms_idx[data[idx][1]]] += 0

                    accu += (cnt / denom)
                accu_list.append(round(accu / epoch, 2))
            elif version == "V2":
                # Testing module V2: In this code-snippet the Greedy policy repeatedly pick the arm with Probability of argmax
                for e in range(epoch):
                    cnt = 0
                    denom = 0
                    for idx in range(s_idx, sz):
                        #Picks the current best arm with probability of it appearing
                        toss = random.random()
                        if toss <= arg_value:
                            cur_arm = self.vizs[arg_max]  # Current best arm
                        else:
                            cur_arm = random.choice(self.vizs) #Picks an arm randomly
                        # pdb.set_trace()
                        denom += 1
                        if data[idx][1] == cur_arm:
                            cnt += 1
                    accu += (cnt / denom)
                accu_list.append(round(accu / epoch, 2))

        return accu_list

    #This following function calculates the total reward acquired by the Greedy policy after running 'runs' time
    def run_regret(self, data, runs):
        sz = len(data)
        ret = []
        for r in range(1, runs + 2):
            model_reward = 0
            if r % 10 == 0: #Using flag to decide when to calculate the predicted rewards
                flag = True
            else:
                flag = False

            for idx in range(sz):
                # self.arms[self.arms_idx[data[idx][1]]] += data[idx][2]
                sum = self.arms.sum()
                self.arms /= sum #For normalizing the rewards accumulated by each arm
                arg_max = np.argmax(self.arms)  # Finds the arm that yielded the maximum reward
                arg_value = self.arms[arg_max]  # Probability
                pred_arm = self.vizs[arg_max]  # Always picking the best action based on past experience.
                # if flag:
                #     pdb.set_trace()
                # print(data[idx][1], pred_arm)
                if data[idx][1] == pred_arm:
                    model_reward += data[idx][2]
                    self.arms[self.arms_idx[data[idx][1]]] += data[idx][2]
                else:
                    self.arms[self.arms_idx[data[idx][1]]] += 0
            # print(r, model_reward)
            if flag:
                ret.append(round(model_reward, 2))
        return ret[len(ret) - 1]
