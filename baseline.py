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
sys.path.insert(0, '/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/')
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import adaptive_epsilon
from mortal_bandit import mortal_bandit
from greedy_policy import Greedy

class baseline:
    def __init__(self):
        path = os.getcwd()
        # self.user_list_faa = glob.glob("D:\\imMens Learning\\Faa_neew\\new_mdp\\*-0ms.xlsx")

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
                    #selection strategy
                    if data[idx][1] == cur_arm and data[idx][2] > 0:
                        pass
                    else:
                        cur_arm = random.choice(self.vizs)
                accu += (cnt / denom)
            # print("Accu = {}".format(round(accu / epoch, 2)))
            accu_list.append(round(accu / epoch, 2))
        return accu_list

    def run_regret(self, data, runs):
        model_reward = 0
        sz = len(data)
        cur_arm = random.choice(self.vizs)  # starting the strategy randomly
        for idx in range(sz):
            if data[idx][1] == cur_arm:
                model_reward += data[idx][2]
            else: #Incorrect guess, pick randomly again
                cur_arm = random.choice(self.vizs)

        # ret = np.ones(int(runs / 10))
        # ret = ret * model_reward
        # return round(ret[len(ret) - 1], 2)
        return round(model_reward, 2)

class plot_accuracy():
    def __init__(self, row, col):
        self.fig, self.ax = plt.subplots(row, col, figsize = (30, 10))
        self.ax = self.ax.flatten()
        self.plt_idx = 0

    def plot(self, user, x, algos, accuracy):
        for idx in range(len(accuracy)):
            # pdb.set_trace()
            for accu in accuracy[idx]:
                accu /= 8
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
        plt.show()
        # plt.savefig(fname, bbox_inches="tight")
        plt.close()

    def sub_plot(self, user, x, algos, accuracy):
        for idx in range(len(accuracy)):
            # pdb.set_trace()
            self.ax[self.plt_idx].plot(x, accuracy[idx])
            # self.ax[self.plt_idx].set(ylabel ='Accuracy', xlabel = 'Threshold')
            self.ax[self.plt_idx].set(ylabel ='Accuracy (Tested on the remaining data after training)', xlabel = '% of data used for Training')
            self.ax[self.plt_idx].set_title('User ' + user)
        self.plt_idx += 1
        # plt.legend(loc='best')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # title = 'Accuracy of various algorithms for different thresholds ' + user
        # plt.title(title)
        # fname = 'figures/' + 'user_' + str(user) + '.png'
        # plt.savefig(fname, bbox_inches="tight")
        # plt.show()

    def finish_subplot(self, algos):
        title = 'Accuracy of various algorithms with varying percentage of % training data'
        self.fig.suptitle(title)
        # plt.ylabel('Accuracy')
        # plt.xlabel('Threshold')
        self.fig.legend(algos)
        # plt.tight_layout()
        # plt.show()
        fname = 'figures/' + 'subplots_4' + '.png'
        plt.savefig(fname, bbox_inches="tight")
        plt.close()


#Going to use the function below, to AGGREGATE the [accuracy vs %training data] for all users
if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wsls = Win_Stay_Lose_Shift(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    # for d in data:
    #     print(wsls.run(d, threshold))
    greedy = Greedy(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    e_greedy_variations = adaptive_epsilon.adaptive_epsilon(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    mortal = mortal_bandit()
    # pltz = plot_accuracy(1, 4)
    idx2 = 0
    algos = []
    wsls_result = 9*[0]
    greedy_result = 9*[0]
    decay_result = 9*[0]
    egreedy_result = 9*[0]
    adaptive_result = 9*[0]
    mortal_result = 9*[0]
    for d in data:
        temp = wsls.run(d, threshold)
        idx = 0
        # pdb.set_trace()
        # print(temp)
        for idx in range(len(temp)):
            wsls_result[idx] += temp[idx]
        # print(temp)
    print(wsls_result)
        # pdb.set_trace()
    #     temp = greedy.run(d, threshold, "V1")
    #     idx = 0
    #     # print(temp)
    #     for idx in range(len(temp)):
    #         greedy_result[idx] += temp[idx]
    #
    #     temp = e_greedy_variations.run_MAB(d, threshold, True)  # Adaptive E-Greedy with decay
    #     idx = 0
    #     # print(temp)
    #     for idx in range(len(temp)):
    #         decay_result[idx] += temp[idx]
    #     temp = e_greedy_variations.run_MAB(d, threshold, False)  # E-Greedy
    #     idx = 0
    #     # print(temp)
    #     for idx in range(len(temp)):
    #         egreedy_result[idx] += temp[idx]
    #     temp = e_greedy_variations.run_MAB_adaptive(d, threshold)
    #     idx = 0
    #     # print(temp)
    #     for idx in range(len(temp)):
    #         adaptive_result[idx] += temp[idx]
    #     temp = mortal.run_stochastic_early_stop(uname[idx2], threshold, d)
    #     idx = 0
    #     # print(temp)
    #     for idx in range(len(temp)):
    #         mortal_result[idx] += temp[idx]
    #     idx2 += 1
    # accu_list = []
    # algos = []
    # accu_list.append(wsls_result)
    # algos.append('Win-Stay-Lose-Shift')
    # accu_list.append(greedy_result)
    # algos.append('Greedy')
    # accu_list.append(decay_result)
    # algos.append('E-Greedy with E-Decay')
    # accu_list.append(egreedy_result)
    # algos.append('E-Greedy')
    # accu_list.append(adaptive_result)
    # algos.append('Adaptive E-Greedy')
    # accu_list.append(mortal_result)
    # algos.append('Mortal Bandit')
    # #Plotting
    # for idx in range(len(accu_list)):
    #     # pdb.set_trace()
    #     for idx2 in range(9):
    #         accu_list[idx][idx2] = round(accu_list[idx][idx2] / 8, 2)
    #     print(algos[idx], accu_list[idx])
    #     plt.plot(threshold, accu_list[idx], label=algos[idx])
    #     plt.ylabel('Accuracy',)
    #     # plt.xticks([])
    #     plt.xlabel('Percent of data used for Training')
    # # plt.legend(loc='best')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # # title = 'Accuracy of various algorithms for different thresholds ' + user
    # # title = "Aggregated"
    # # plt.title(title)
    # # plt.show()
    # fname = 'figures/' + 'aggregated' + '.png'
    # plt.savefig(fname, bbox_inches="tight")
    # plt.close()

#The following main function produces a [Accuracy vs %training data]graph for all users seperately
# if __name__ == "__main__":
#     obj = integrate()
#     data, uname = obj.get_files()
#     threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     wsls = Win_Stay_Lose_Shift(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
#     # for d in data:
#     #     print(wsls.run(d, threshold))
#     greedy = Greedy(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
#     e_greedy_variations = adaptive_epsilon.adaptive_epsilon(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
#     mortal = mortal_bandit()
#     pltz = plot_accuracy(1, 4)
#     idx = 0
#     algos = []
#     for d in data:
#         # print(wsls.run(d, threshold))
#         # print(greedy.run(d, threshold))
#         accu_list = []
#         algos = []
#         wsls_result = wsls.run(d, threshold)
#         accu_list.append(wsls_result)
#         algos.append('Win-Stay-Lose-Shift')
#         greedy_result = greedy.run(d, threshold, "V1")
#         accu_list.append(greedy_result)
#         algos.append('Greedy')
#         # greedy_result = greedy.run(d, threshold, "V2")
#         # accu_list.append(greedy_result)
#         # algos.append('Greedy V2')
#         decay_result = e_greedy_variations.run_MAB(d, threshold, True)  # Adaptive E-Greedy with decay
#         accu_list.append(decay_result)
#         algos.append('E-Greedy with E-Decay')
#         egreedy_result = e_greedy_variations.run_MAB(d, threshold, False)  # E-Greedy
#         algos.append('E-Greedy')
#         accu_list.append(egreedy_result)
#         adaptive_result = e_greedy_variations.run_MAB_adaptive(d, threshold)
#         algos.append('Adaptive E-Greedy')
#         accu_list.append(adaptive_result)
#         algos.append('Mortal Bandit')
#         mortal_result = mortal.run_stochastic_early_stop(threshold, d)
#         accu_list.append(mortal_result)
#         pltz.sub_plot(uname[idx], threshold, algos, accu_list)
#         idx += 1
#     pltz.finish_subplot(algos)

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