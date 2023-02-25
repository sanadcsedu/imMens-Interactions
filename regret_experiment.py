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
from adaptive_epsilon import adaptive_epsilon
from mortal_bandit import mortal_bandit
from greedy_policy import Greedy
from baseline import Win_Stay_Lose_Shift

class regret_experiment:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 4)
        self.bar_width = .2

    def get_optimal_reward(self, data, runs):
        ret_reward = 0
        sz = len(data)
        for idx in range(sz):
            ret_reward += data[idx][2]
        # ret = np.ones(int(runs / 10))
        # ret = ret * ret_reward
        # return ret
        return ret_reward

    def sub_plot(self, idx, user, results, algos):
        x = np.arange(len(results))
        colors = ['red', 'green', 'blue', 'black', 'yellow']
        self.ax[idx].bar(x, results, color = colors)
        # self.ax[self.plt_idx].set(ylabel ='Accuracy', xlabel = 'Threshold')
        # self.ax[self.plt_idx].set(ylabel='Accuracy (Tested on the remaining data after training)',
        #                           xlabel='% of data used for Training')
        self.ax[idx].set_title('User ' + user)

    def plot(self, algos):
        title = 'Regret'
        self.fig.suptitle(title)
        # plt.ylabel('Accuracy')
        # plt.xlabel('Threshold')
        colors = ['red', 'green', 'blue', 'black', 'yellow']
        self.fig.legend(algos, colors)
        z = np.arange(len(algos))
        plt.xticks(z, algos)
        # plt.tight_layout()
        plt.show()
        # fname = 'figures/' + 'subplots_2' + '.png'
        # plt.savefig(fname, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    optimal = regret_experiment()
    greedy = Greedy(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    WSLS = Win_Stay_Lose_Shift(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    epsilon_variations = adaptive_epsilon(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    mortal = mortal_bandit()
    idx = 0
    runz = 50
    algos = []
    for d in data:
        algos = []
        results = np.array([])
        optimal_reward = optimal.get_optimal_reward(d, runz)
        greedy_reward = greedy.run_regret(d, runz)
        results = np.append(results, greedy_reward)
        algos.append('Greedy')
        WSLS_reward = WSLS.run_regret(d, runz)
        results = np.append(results, WSLS_reward)
        algos.append('Win-Stay-Lose-Shift')
        epsilon_greedy = epsilon_variations.run_MAB_regret(d, runz, False)
        results = np.append(results, epsilon_greedy)
        algos.append('E-Greedy')
        epsilon_decay = epsilon_variations.run_MAB_regret(d, runz, True)
        results = np.append(results, epsilon_decay)
        algos.append('E-Greedy with E-Decay')
        adaptive_epsilon = epsilon_variations.run_MAB_Adaptive_regret(d, runz)
        results = np.append(results, adaptive_epsilon)
        algos.append('Adaptive E-Greedy')
        mortal_reward = mortal.stochastic_early_stop_regret(uname[idx], d, runz)
        results = np.append(results, mortal_reward)
        algos.append('Mortal Bandit')
        # print(results)
        results = results / optimal_reward
        print(uname[idx])
        for i in results:
            print(round(i, 2), end= ", ")
        print()
        # optimal.sub_plot(idx, uname[idx], results, algos)
        idx += 1
    # optimal.plot(algos)
    print(algos)