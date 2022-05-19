import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd

class environment2:
    def __init__(self):
        path = os.getcwd()
        self.user_list_bright = glob.glob(path + "/QLearning/*_reformed.csv")
        self.user_list_faa = glob.glob(path + "/QLearning/*_reform.csv")
        
        # self.user_list_bright = glob.glob('D:\\imMens Learning\\QLearning\\*_reformed.csv')
        # self.user_list_faa = glob.glob('D:\\imMens Learning\\QLearning\\*_reform.csv')
        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.valid_actions = ["same", "change"]
        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0
        self.prev_state = None

    def reset(self, all=False, test = False):
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

    def get_state(self, state):
        state = state.strip('()')
        state = state.replace(" ", "")
        # state = state.split('+')[0]
        return state

    # Optimization is not the priority right now
    # Returns all interactions for a specific user and subtask i.e. user 'P1', subtask '1.txt'
    def process_data(self, filename, thres):
        # df = pd.read_excel(filename, sheet_name= "Sheet1", usecols="B:D")
        df = pd.read_csv(filename)
        prev_state = None
        cnt_inter = 0
        for index, row in df.iterrows():
            # print(row)
            # pdb.set_trace()
            # print("here {} end\n".format(cnt_inter))
            cur_state = self.get_state(row['State'])
            # if cur_state not in ('Question', 'Sensemaking'):
            #     continue
            if prev_state == cur_state:
                action = "same"
            else:
                action = "change"
            self.mem_states.append(cur_state)
            self.mem_reward.append(row['reward'])
            self.mem_action.append(action)
            cnt_inter += 1
            prev_state = cur_state
        self.threshold = int(cnt_inter * thres)
        # print("{} {} {}\n".format(cnt_inter, len(self.mem_states), self.threshold))

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test = False):
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
        # pdb.set_trace()
        # if test:
            # print("{} {}".format(self.valid_actions[act_arg], cur_action))
            # x, y, z = self.cur_inter(self.steps)
            # print("{} {} {} {}".format(self.steps, x, y, z))
        if self.valid_actions[act_arg] == cur_action:
            prediction = 1
        else:
            prediction = 0
        self.take_step_action(test)
        return next_state, cur_reward, self.done, prediction


if __name__ == "__main__":
    env = environment2()
    # users = env.user_list
    # print(users)
    env.process_data('D:\imMens Learning\QLearning\p10_reform.csv', 0)
    # for idx in range(len(env.mem_states)):
    #     print("{} {}".format(env.mem_states[idx], env.mem_reward[idx]))
    # print(users)


