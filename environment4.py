# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd


class environment4:
    def __init__(self):
        path = os.getcwd()
        # self.user_list_bright = glob.glob(path + "/QLearning/*_reformed.csv")
        self.user_list_faa = glob.glob("D:\\imMens Learning\\Faa_neew\\*-0ms.xlsx")

        # self.user_list_bright = glob.glob('D:\\imMens Learning\\QLearning\\*_reformed.csv')
        # self.user_list_faa = glob.glob('D:\\imMens Learning\\QLearning\\*_reform.csv')
        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        # self.valid_actions = ["same", "change"]
        self.valid_actions = ["observation", "steer", "explanation", "generalization"]
        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0
        self.prev_state = None
        # self.find_states = defaultdict()

    def reset(self, all=False, test=False):
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
            # print("start {}".format(self.steps))
            # pdb.set_trace()
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

    # In this function the states combination of visualizations + (sensemaking, question)
    # def get_state(self, state):
    #     state = state.strip('()')
    #     state = state.replace(" ", "")
    #     # state = state.split('+')[0]
    #     self.find_states[state] = 1
    #     return state

    # The states are now temporal vs spatial data + (sensemaking, question)
    def get_state(self, state):
        state = state.strip('()')
        state = state.replace(" ", "")
        state = state.split('+')
        if state[1] in ["scatterplot-0-1"]:
            new_state = state[0] + "+spatial"
            # new_state = "spatial"
        elif state[1] in ["bar-4"]:
            new_state = state[0] + "+carrier"
            # new_state = "carrier"
        else:
            new_state = state[0] + "+temporal"
            # new_state = "temporal"

        # pdb.set_trace()
        # print(state, new_state)
        # self.find_states[new_state] = 1
        return new_state

    # Optimization is not the priority right now
    # Returns all interactions for a specific user and subtask i.e. user 'P1', subtask '1.txt'
    def process_data(self, filename, thres):
        # df = pd.read_excel(filename, sheet_name= "Sheet1", usecols="B:D")
        df = pd.read_excel(filename, sheet_name="Sheet(2)", usecols="A:E, I:J")
        cnt_inter = 0
        for index, row in df.iterrows():
            self.mem_states.append(row['State'])
            self.mem_reward.append(row['Reward'])
            self.mem_action.append(row['Action'])
            # print(row['State'], row['action'])
            cnt_inter += 1
        self.threshold = int(cnt_inter * thres)
        # print("{} {} {}\n".format(cnt_inter, len(self.mem_states), self.threshold))

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test=False):
        # pdb.set_trace()
        if test:
            if len(self.mem_states) > self.steps + 1:
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
        # pdb.set_trace()

        self.take_step_action(test)

        return next_state, cur_reward, self.done, prediction
