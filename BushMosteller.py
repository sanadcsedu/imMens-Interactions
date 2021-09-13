import numpy as np
from collections import defaultdict
import pdb


class BushMosteller:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.prob = defaultdict(float)

    # User already has some prior knowledge about some strategies regarding the given task
    def add_prior_strategies(self, priors):
        for attr in priors:
            self.prob[attr] = (1 - self.prob[attr]) * self.alpha
        self.prob[attr] -= self.prob[attr] * self.beta

    def update(self, action, r):
        if r > 0:
            self.prob[action] += (1 - self.prob[action]) * self.alpha
        else:
            self.prob[action] -= self.prob[action] * self.beta

        for keys in self.prob:
            if keys != action:
                if r > 0:
                    self.prob[keys] -= self.prob[keys] * self.alpha
                else:
                    self.prob[keys] += (1 - self.prob[keys]) * self.beta

    def normalize(self):
        sum = 0
        for attr in self.prob:
            sum += self.prob[attr]
        if sum > 0:
            for attr in self.prob:
                self.prob[attr] /= sum

    #This method selects action proportional to probablity distribution
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

    def make_choice_nostate(self):
        self.normalize()
        prob = self.prob.copy()
        return self.select_from_ptable(prob)

    def random_selection(self):
        choices = list(self.prob.keys())
        pick = np.random.randint(0, len(choices))
        return choices[pick]