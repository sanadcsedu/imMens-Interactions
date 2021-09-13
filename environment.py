import os
import fnmatch
import pdb
from collections import defaultdict

class environment:
    def __init__(self):
        #Contains list of the users (sub-directories) from the "Brightkite-state-subtask" folder
        self.user_list = os.listdir('Brightkite-state-subtask/')

        # Going inside the "Brightkite-state-subtask" directory
        self.cur_dir = os.path.join(os.getcwd(), 'Brightkite-state-subtask/')
        # self.cur_dir = os.path.abspath(os.getcwd())

        #For testing if things are working properly
        self.actions = defaultdict()
        self.states = defaultdict()

        #Valid actions and states we are going to use
        self.valid_actions = ["'pan'", "'zoom'", "'brush'", "'range select'"]
        self.valid_states = ["'Exploration'", "'Question / Hypothesis'", "'Sensemaking'", "'Drill-Down'"]

    #Optimization is not the priority right now
    def get_user(self):
        #going into each users' folder and reading their interactions
        for sub_dir in self.user_list:
            # print(sub_dir)
            dest = os.path.join(self.cur_dir, sub_dir)
            subtasks = os.listdir(dest)
            for s in subtasks:
                self.get_subtasks(dest, s)

    def is_valid(self, action, state):
        if action not in self.valid_actions:
            return False
        if state not in self.valid_states:
            return False
        return True

    #Returns all interactions for a specific user and subtask i.e. user 'P1', subtask '1.txt'
    def get_subtasks(self, path, file):
        filename = os.path.join(path, file)
        data = open(filename, 'r')
        # print(file)
        for lines in data:
            lines = lines.strip('[]\n')
            interaction = lines.split(', ')

            action = self.get_action(interaction)
            state = self.get_state(interaction)
            if not self.is_valid(action, state):
                continue
            # pdb.set_trace()
            self.actions[self.get_action(interaction)] = True
            self.states[self.get_state(interaction)] = True

        data.close()
        return data

    #Returns the state from the interaction
    def get_state(self, interaction):
        sz = len(interaction)
        return interaction[sz - 1]

    #Returns the action performed in the interaction
    def get_action(self, interaction):
        # sz = len(interaction)
        # print("{} {}".format(sz, interaction))
        return interaction[2]


if __name__ == "__main__":
    env = environment()
    env.get_user()
    print(env.actions.keys())
    print(env.states.keys())
