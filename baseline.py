# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import misc

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

if __name__ == "__main__":
    obj = baseline()
    users = obj.user_list_faa
    accu_list = []
    misc_obj = misc.misc(users)
    res = []
    name = []
    threshold_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for u in users:
        obj.process_data(obj.user_list_faa[0], 1)
        sz = len(obj.mem_states)
        # print(sz)
        # for idx in range(sz-1):
        #     print("{} {} {}".format(obj.mem_states[idx], obj.mem_action[idx], obj.mem_reward[idx]))
        #The baseline model always picks the action 'switch', because the user wants to use information from multiple visualization
        correct = 0
        res = []
        for thres in threshold_h:
            limit = int(thres * (sz - 1))
            correct = 0
            denom = 0
            for idx in range(limit, sz - 1):
                if obj.mem_action[idx] == 'switch':
                    correct += 1
                denom += 1
            accu = correct / (denom)
            res.append(round(accu, 2))
        accu_list.append(res)
        name.append(misc_obj.get_user_name(u))
    pdb.set_trace()
    misc_obj.plot_together(accu_list, name, 'baseline')