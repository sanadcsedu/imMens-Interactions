# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd


class environment5:
    def __init__(self):
        path = os.getcwd()
        self.user_list_faa = glob.glob("D:\\imMens Learning\\Faa_neew\\p6-faa-0ms.xlsx")

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


    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test=False):
        if test:
            if len(self.mem_states) > self.steps + 3:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0
        else:
            if self.threshold > self.steps + 1:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0

    # act_arg = action argument refers to action number
    def step(self, cur_state, act_arg, test):
        _, cur_reward, cur_action = self.cur_inter(self.steps)
        _, temp_step = self.peek_next_step()
        next_state, next_reward, next_action = self.cur_inter(temp_step)
        # if test:
        #     print("{} {}".format(self.valid_actions[act_arg], cur_action))
        #     x, y, z = self.cur_inter(self.steps)
        #     print("{} {} {} {}".format(self.steps, x, y, z))
        if self.valid_actions[act_arg] == cur_action:
            prediction = 1
        else:
            prediction = 0

        self.take_step_action(test)

        return next_state, cur_reward, self.done, prediction

# if __name__ == "__main__":
#     obj = environment5()
#     obj.process_data(obj.user_list_faa[0], 1)
#     sz = len(obj.mem_states)
#     print(sz)
#     for idx in range(sz-1):
#         print("{} {} {}".format(obj.mem_states[idx], obj.mem_action[idx], obj.mem_reward[idx]))
#
