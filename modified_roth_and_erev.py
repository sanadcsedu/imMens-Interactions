import numpy as np
from collections import defaultdict
from collections import Counter
import pdb

class modified_roth_and_erev:

    def __init__(self):
        self.cutoff = None
        self.q_value = defaultdict(float)
        self.prob = defaultdict(float)

    def add_prior_strategies(self, priors, payoff):
        for actions in priors:
            self.prob[actions] = 0
        for actions in priors:
            self.q_value[actions] += payoff
        # pdb.set_trace()

    #the Q values are updated using the following function and probablities are immediately calculated
    def update_qtable(self, action, payoff, forgetting):
        #Updating Q-values for Pure Strategies similar to Basic Roth and Erev
        self.q_value[action] += payoff
        #Introducing the Forgetting parameter
        for strategies in self.q_value:
            self.q_value[strategies] *= (1 - forgetting)
        # pdb.set_trace()

    #Get the probability of each strategy from the Q values
    def normalize(self):
        sum = 0
        for actions in self.q_value:
            sum += self.q_value[actions]
        #Normalizing
        if sum > 0:
            for actions in self.q_value:
                self.prob[actions] = self.q_value[actions] / sum

    def select_from_ptable(self, temp_prob):
        cur_max = -1
        ret = []
        for attr in temp_prob:
            if cur_max < temp_prob[attr]:
                cur_max = temp_prob[attr]
                ret = []
                ret.append(attr)
            elif cur_max == temp_prob[attr]:
                ret.append(attr)

        picked = np.random.randint(0, len(ret))
        best_action = ret[picked]

        threshold = np.random.random()
        if threshold > cur_max:
            best_action = self.random_selection()
        # pdb.set_trace()
        # print("{} {} {}".format(threshold, cur_max, threshold > cur_max))
        return best_action

    def random_selection(self):

        choices = list(self.prob.keys())
        # pdb.set_trace()
        pick = np.random.randint(0, len(choices))
        return choices[pick]

    def make_choice_nostate(self):
        self.normalize()
        temp_prob = self.prob.copy()
        return self.select_from_ptable(temp_prob)
