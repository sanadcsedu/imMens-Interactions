# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import misc
import random
import sys
sys.path.insert(0, "D:\\imMens Learning\\stationarity_test")
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import adaptive_epsilon

class baseline:
    def __init__(self):
        path = os.getcwd()
        self.user_list_faa = glob.glob("D:\\imMens Learning\\Faa_neew\\new_mdp\\*-0ms.xlsx")

        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.valid_actions = ["stay", "switch"]
        self.valid_states = ["scatterplot", "year", "month", "carrier"]
        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0
        self.prev_state = None
        self.find_states = defaultdict()

    def get_state(self, state):
        states = state.split(' ')
        return states

    def process_data(self, filename, thres):
        # pdb.set_trace()
        df = pd.read_excel(filename, sheet_name="Sheet3", usecols="B:C")
        cnt_inter = 0
        for index, row in df.iterrows():
            states = self.get_state(row['State'])
            for s in states:
                if s in self.valid_states:
                    self.mem_states.append(s)
                    self.mem_reward.append(row['Reward'])
            cnt_inter += 1
        length = len(self.mem_states)
        for idx in range(length - 1):
            if self.mem_states[idx] != self.mem_states[idx + 1]:
                self.mem_action.append('switch')
            else:
                self.mem_action.append('stay')

        self.threshold = int(cnt_inter * thres)


class Win_Stay_Lose_Shift():
    def __init__(self, vizs):
        self.vizs = vizs

    def run(self, data, threshold):
        epoch = 10
        accu_list = []
        for thres in threshold:
            sz = len(data)
            s_idx = int(thres * sz)
            accu = 0
            for e in range(epoch):
                cnt = 0
                cur_arm = random.choice(self.vizs)
                denom = 0
                for idx in range(s_idx, sz):
                    denom += 1
                    if data[idx][1] == cur_arm:
                        cnt += 1
                    else:
                        cur_arm = random.choice(self.vizs)
                accu += (cnt / denom)
            # print("Accu = {}".format(round(accu / epoch, 2)))
            accu_list.append(round(accu / epoch, 2))
        return accu_list

class Greedy():
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
                self.arms[self.arms_idx[data[idx][1]]] += 1
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
                    cur_arm = self.vizs[arg_max] #Always picking the best action based on past experience.
                    denom = 0
                    for idx in range(s_idx, sz):
                        denom += 1
                        if data[idx][1] == cur_arm:
                            cnt += 1
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

class plot_accuracy():
    def __init__(self):
        pass

    def plot(self, user, x, algos, accuracy):
        for idx in range(len(accuracy)):
            # pdb.set_trace()
            plt.plot(x, accuracy[idx], label = algos[idx])
            plt.ylabel('Accuracy')
            # plt.xticks([])
            plt.xlabel('Threshold')
        # plt.legend(loc='best')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # title = 'Accuracy of various algorithms for different thresholds ' + user
        title = user
        plt.title(title)
        # plt.show()
        fname = 'figures/' + 'user_' + str(user) + '.png'
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    def subplot(self, user, x, algos, accuracy):
        for idx in range(len(accuracy)):
            # pdb.set_trace()
            plt.plot(x, accuracy[idx], label = algos[idx])
            plt.ylabel('Accuracy')
            # plt.xticks([])
            plt.xlabel('Threshold')
        # plt.legend(loc='best')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        title = 'Accuracy of various algorithms for different thresholds ' + user
        plt.title(title)
        # plt.show()
        fname = 'figures/' + 'user_' + str(user) + '.png'
        plt.savefig(fname, bbox_inches="tight")

if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wsls = Win_Stay_Lose_Shift(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    # for d in data:
    #     print(wsls.run(d, threshold))
    greedy = Greedy(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    e_greedy_variations = adaptive_epsilon.adaptive_epsilon(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])

    pltz = plot_accuracy()
    idx = 0
    for d in data:
        # print(wsls.run(d, threshold))
        # print(greedy.run(d, threshold))
        accu_list = []
        algos = []
        wsls_result = wsls.run(d, threshold)
        accu_list.append(wsls_result)
        algos.append('Win-Stay-Lose-Shift')
        greedy_result = greedy.run(d, threshold, "V1")
        accu_list.append(greedy_result)
        algos.append('Greedy V1')
        # greedy_result = greedy.run(d, threshold, "V2")
        # accu_list.append(greedy_result)
        # algos.append('Greedy V2')
        decay_result = e_greedy_variations.run_MAB(d, threshold, True)  # Adaptive E-Greedy with decay
        accu_list.append(decay_result)
        algos.append('E-Greedy with E-Decay')
        egreedy_result = e_greedy_variations.run_MAB(d, threshold, False)  # E-Greedy
        algos.append('E-Greedy')
        accu_list.append(egreedy_result)
        adaptive_result = e_greedy_variations.run_MAB_adaptive(d, threshold)
        algos.append('Adaptive E-Greedy')
        accu_list.append(adaptive_result)
        pltz.plot(uname[idx], threshold, algos, accu_list)
        idx += 1

# if __name__ == "__main__":
#     obj = baseline()
#     users = obj.user_list_faa
#     accu_list = []
#     misc_obj = misc.misc(users)
#     res = []
#     name = []
#     threshold_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     for u in users:
#         obj.process_data(obj.user_list_faa[0], 1)
#         sz = len(obj.mem_states)
#         # print(sz)
#         # for idx in range(sz-1):
#         #     print("{} {} {}".format(obj.mem_states[idx], obj.mem_action[idx], obj.mem_reward[idx]))
#         #The baseline model always picks the action 'switch', because the user wants to use information from multiple visualization
#         correct = 0
#         res = []
#         for thres in threshold_h:
#             limit = int(thres * (sz - 1))
#             correct = 0
#             denom = 0
#             for idx in range(limit, sz - 1):
#                 if obj.mem_action[idx] == 'switch':
#                     correct += 1
#                 denom += 1
#             accu = correct / (denom)
#             res.append(round(accu, 2))
#         accu_list.append(res)
#         name.append(misc_obj.get_user_name(u))
#     pdb.set_trace()
#     misc_obj.plot_together(accu_list, name, 'baseline')