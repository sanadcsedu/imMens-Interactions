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

        #This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.substasks = None
        self.cur_subtask = 0
        self.done = False # Done exploring the current subtask
        self.complete = False #Flag marks the end of all interactions for a user (all subtasks)
        
        #Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_actions = []
        self.mem_states = []

    def reset(self):
        #Resetting the variables used for tracking position of the agents
        self.steps = 0
        self.substasks = None
        self.cur_subtask = 0
        self.done = False
        self.complete = False

        self.mem_actions = []
        self.mem_states = []


    #Optimization is not the priority right now
    def get_user(self, user):
        #going into each users' folder and reading their interactions
        for sub_dir in self.user_list:
            if sub_dir != user:
                continue
            # print(sub_dir)
            dest = os.path.join(self.cur_dir, sub_dir)
            self.subtasks = sorted(os.listdir(dest))
            for s in self.subtasks:
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
        action = []
        state = []
        for lines in data:
            lines = lines.strip('[]\n')
            interaction = lines.split(', ')

            temp_act = self.get_action(interaction)
            temp_state = self.get_state(interaction)
            if not self.is_valid(temp_act, temp_state):
                continue
            action.append(temp_act)
            state.append(temp_state)
            # pdb.set_trace()
            # self.actions[self.get_action(interaction)] = True
            # self.states[self.get_state(interaction)] = True
        data.close()
        self.mem_actions.append(action)
        self.mem_states.append(state)

    #Returns the state from the interaction
    def get_state(self, interaction):
        sz = len(interaction)
        return interaction[sz - 1]

    #Returns the action performed in the interaction
    def get_action(self, interaction):
        # sz = len(interaction)
        # print("{} {}".format(sz, interaction))
        return interaction[2]

    def cur_inter(self):
        return self.mem_states[self.cur_subtask][self.steps], self.mem_actions[self.cur_subtask][self.steps]

    def take_step_action(self):
        if len(self.mem_actions[self.cur_subtask]) > self.steps + 1:
            self.steps += 1
        else:
            self.done = True
            self.steps = 0
            
    def take_step_subtask(self):
        if len(self.mem_actions) > self.cur_subtask + 1:
            self.cur_subtask += 1
        else:
            self.complete = True
            self.cur_subtask = 0
        self.done = False
        #pdb.set_trace()
            
if __name__ == "__main__":
    env = environment()
    users = env.user_list
    # print(users)
    env.get_user(users[3])
    #cnt = 0
    while(not env.complete):
        print("SUBTASK: {}".format(env.cur_subtask))
        while(not env.done):
            a, s = env.cur_inter()
            print(env.steps, a, s)
            env.take_step_action()
            #cnt += 1
        env.take_step_subtask()

    # print(env.subtasks)
    #for i in range(6):
    #    print(len(env.mem_actions[i]))
    # print(env.actions.keys())
    # print(env.states.keys())
