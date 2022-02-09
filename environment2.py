import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd

class environment2:
    def __init__(self):
        self.user_list = glob.glob('D:\\imMens Learning\\Brightkite_neew\\joined\\*_reformed.csv')
        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.valid_actions = ["keep", "change"]
        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.prev_state = None

    def reset(self, all=False):
        # Resetting the variables used for tracking position of the agents
        self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
        s, r = self.cur_inter(self.steps)
        return s

    # Optimization is not the priority right now
    # Returns all interactions for a specific user and subtask i.e. user 'P1', subtask '1.txt'
    def get_subtasks(self, filename):
        # df = pd.read_excel(filename, sheet_name= "Sheet1", usecols="B:D")
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            self.mem_states.append(row['State'])
            self.mem_reward.append(row['reward'])

    # Returns the state from the interaction
    def get_state(self, interaction):
        sz = len(interaction)
        return interaction[sz - 1]

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self):
        # pdb.set_trace()
        if len(self.mem_states) > self.steps + 1:
            self.steps += 1
        else:
            self.done = True
            self.steps = 0

    # act_arg = action argument refers to action number
    def step(self, cur_state, act_arg):
        _, temp_step = self.peek_next_step()
        ground_next_state, _ = self.cur_inter(temp_step)
        _, cur_reward = self.cur_inter(temp_step)
        if act_arg == "change":
            if cur_state == "Question":
                next_state = "Sensemaking"
            else:
                next_state = "Question"
        else:
            next_state = cur_state
        # if next_state == ground_next_state:
        #     cur_reward += 1
        self.take_step_action()
        return ground_next_state, cur_reward, self.done


if __name__ == "__main__":
    env = environment2()
    users = env.user_list
    print(users)
    # env.get_subtasks(users[0])
    # for idx in range(len(env.mem_states)):
    #     print("{} {}".format(env.mem_states[idx], env.mem_reward[idx]))
    # print(users)