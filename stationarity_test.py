# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import matplotlib.pyplot as plt


class stationarity_test:
    def __init__(self):
        path = os.getcwd()
        self.user_list_faa = glob.glob("D:\\imMens Learning\\Faa_neew\\p2-faa-0ms.xlsx")

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

    def reset(self, all=False, test=False):
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s

    # The states are now temporal vs spatial data + (sensemaking, question)
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

if __name__ == "__main__":
    obj = stationarity_test()
    obj.process_data(obj.user_list_faa[0], 1)
    length = len(obj.mem_states)
    transition = defaultdict(lambda: defaultdict(float))
    transition_list = defaultdict(lambda: defaultdict(list))
    transition_tuple = defaultdict(list)
    s = ['scatterplot', 'carrier', 'month', 'year']
    states = []
    for a in s:
        for b in s:
            states.append((a, b))
    print(states)
    pdb.set_trace()
    # return

    for a in states:
        for b in states:
            transition[a][b] = 1
            # transition_list[a][b].append(1)
            transition_list[a][b].append(0.25)

    for i in range(length - 1):
        transition[obj.mem_states[i]][obj.mem_states[i+1]] += 1
        sum = 0
        for keys2 in transition[obj.mem_states[i]].keys():
            sum += transition[obj.mem_states[i]][keys2]
        for s in states:
            transition_list[obj.mem_states[i]][s].append(round(transition[obj.mem_states[i]][s] / sum, 2))
            # transition_list[obj.mem_states[i]][obj.mem_states[i+1]].append(round(transition[obj.mem_states[i]][obj.mem_states[i+1]] / sum, 2))

        temp = []
        for s in states:
            temp.append(round(transition[obj.mem_states[i]][s] / sum, 2))
        transition_tuple[obj.mem_states[i]].append(temp)

    # for keys1 in transition_tuple:
    #     idx = 0
    #     for items in transition_tuple[keys1]:
    #         print(idx, items)
    #         idx += 1

    # for keys1 in transition:
    #     print("{} ".format(keys1), end = " ")
    #     for keys2 in transition[keys1]:
    #         print("{} {} ".format(keys2, transition[keys1][keys2]), end=" ")
    #     print("\n")
    #
    for keys1 in transition:
        # print("{} ".format(keys1), end = " ")
        for keys2 in transition[keys1]:
            # print("{} {} ".format(keys2, transition_list[keys1][keys2]), end=" ")
            plt.plot(transition_list[keys1][keys2], label = keys2)
            plt.ylabel('Probability')
            plt.xticks([])
            plt.xlabel('time')
        plt.legend(loc='best')
        title = 'First Visualization ' + keys1
        plt.title(title)
        plt.show()
        # print("\n")