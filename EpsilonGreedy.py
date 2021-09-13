import pdb
import numpy as np
from _collections import defaultdict
import math

#Includes both classical E-Greedy and Adaptive E-Greedy algorithm
class EpsilonGreedy:
    def __init__(self, all_attrs, epsilon, l, f):
        self.strategies = defaultdict(list,{key: 0 for key in all_attrs})
        self.q = defaultdict(list,{key: 0 for key in all_attrs})
        self.epsilon = epsilon
        self.t = 0 #Keeps track of how many times the algorithm performs exploration
        self.l = l #maximum number of time exploration can continue
        self.f = f #Regularization Parameter
        self.max_prev = 0
        self.max_cur = 0
        self.cnt = 0


    def make_choice_classic(self):

        if np.random.random() > self.epsilon:
            #return K strategies with maximum probabilities
            ret = self.best_selection()
        else:
            #return k strategies Randomly
            ret = self.random_selection()
        return ret

    def make_choice_adaptive(self):

        if np.random.random() > self.epsilon:
            #return K strategies with maximum probabilities
            ret = self.best_selection()
        else:
            #return k strategies Randomly and see if you need to change the value of Epsilon
            self.t += 1
            if self.t == self.l and self.cnt > 0:
                self.max_cur /= self.cnt
                delta = ((self.max_cur - self.max_prev) * self.f)
                self.change_epsilon(delta)
                self.max_prev = self.max_cur
                self.max_cur = 0
                self.cnt = 0
                self.t = 0

            ret = self.random_selection()

        return ret

    def change_epsilon(self, delta):
        if delta > 0:
            self.epsilon = 1 / (1 + math.exp(-2 * delta))
            self.epsilon -= 0.5
            self.epsilon /= 2
        else:
            self.epsilon = 0.05

    def update(self, action, reward):
        self.max_cur += reward
        self.cnt += 1

        self.strategies[action] += 1
        n = self.strategies[action]
        self.q[action] += self.q[action] * ((n - 1) / n) + (reward / n)

    # This method selects action proportional to probablity distribution
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

    def best_selection(self):
        prob = self.q.copy()
        return self.select_from_ptable(prob)

    def random_selection(self):
        choices = list(self.strategies.keys())
        pick = np.random.randint(0, len(choices))
        return choices[pick]
