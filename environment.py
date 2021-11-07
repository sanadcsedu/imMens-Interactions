import os
import fnmatch
import pdb
from collections import defaultdict

class environment:
    def __init__(self):
        #Contains list of the users (sub-directories) from the "Brightkite-state-subtask" folder
        self.user_list = os.listdir('temp/')

        # Going inside the "Brightkite-state-subtask" directory
        self.cur_dir = os.path.join(os.getcwd(), 'temp/')
        # self.cur_dir = os.path.abspath(os.getcwd())

        #For testing if things are working properly
        self.actions = defaultdict()
        self.states = defaultdict()

        #Valid actions and states we are going to use
        self.valid_actions = ["'pan'", "'zoom'", "'brush'", "'range select'"]
        self.valid_states = ["'Exploration'", "'Question'", "'Sensemaking'", "'Interface-feedback'"]

        #This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.substasks = None
        self.cur_subtask = 0
        self.done = False # Done exploring the current subtask
        self.complete = False #Flag marks the end of all interactions for a user (all subtasks)
        
        #Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_actions = []
        self.mem_states = []

    def reset(self, all = False):
        #Resetting the variables used for tracking position of the agents
        self.steps = 0
        self.substasks = None
        self.cur_subtask = 0
        self.done = False
        self.complete = False
        if all:
            self.mem_actions = []
            self.mem_states = []
        s, a = self.cur_inter(self.cur_subtask, self.steps)
        return s

    #Optimization is not the priority right now
    def get_user(self, user):
        #going into each users' folder and reading their interactions
        for sub_dir in self.user_list:
            if sub_dir != user:
                continue
            print(sub_dir)
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

    def cur_inter(self, cur_subtask, steps):
        return self.mem_states[cur_subtask][steps], self.mem_actions[cur_subtask][steps]

    def peek_next_step(self):
        if len(self.mem_actions[self.cur_subtask]) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self):
        # pdb.set_trace()
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

    def gen_reward(self, selected_action, ground_action):
        if selected_action == ground_action:
            return 1
        else:
            return -1

    #act_arg = action argument refers to action number
    def step(self, cur_state, act_arg, train = True):
        temp_done, temp_step = self.peek_next_step()
        ground_next_state, ground_action = self.cur_inter(self.cur_subtask, temp_step)
        reward = self.gen_reward(self.valid_actions[act_arg], ground_action)
        # print("{} {} {} {}".format(cur_state, self.valid_actions[act_arg], ground_action, reward))
        # pdb.set_trace()
        if train: #While training MDP
            if reward > 0:
                self.take_step_action()
                next_state = ground_next_state
            else:
                next_state = cur_state
            # print(self.steps)
        else: #While testing the MDP
            self.take_step_action()
            next_state = ground_next_state

        return next_state, reward, self.done

if __name__ == "__main__":
    env = environment()
    users = env.user_list
    print(users)
    _map = defaultdict(lambda: defaultdict(int))
    for u in users:
        env.get_user(u)
        # cnt = 0
        while (not env.complete):
            # print("SUBTASK: {}".format(env.cur_subtask))
            while (not env.done):
                s, a = env.cur_inter()
                # pdb.set_trace()
                _map[s][a] += 1
                # print(env.steps, a, s)
                env.take_step_action()
                # cnt += 1
            env.take_step_subtask()
        env.reset()
    # print(_map)
    for states in _map.keys():
        print("{}".format(states), end=" ")
        sum = 0
        for actions in _map[states].keys():
            # print("{} {} ".format(actions, _map[states][actions]), end=" ")
            sum += _map[states][actions]
        for actions in _map[states].keys():
            print("{} {:.2f}% ".format(actions, _map[states][actions]*100 / sum), end=" ")
        print("")
    # print(env.subtasks)
    #for i in range(6):
    #    print(len(env.mem_actions[i]))
    # print(env.actions.keys())
    # print(env.states.keys())
