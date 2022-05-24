#contains all the miscellaneous functions for running 
import pdb
import TDLearning
import TD_SARSA 
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import sys
import plotting
import environment2
from tqdm import tqdm

class misc:
    def __init__(self):
        self.discount_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # discount_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        # alpha_h = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.alpha_h = [0.1, 0.2, 0.3, 0.4, .5]
        # epsilon_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.epsilon_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.threshold_h = [0.75]
        # threshold_h = [0.4, 0.6, 0.7, 0.8, 0.9]
        self.prog = 4 * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h)
    
    def get_user_name(self, url):
        string = url.split('/')
        fname = string[len(string) - 1]
        uname = fname.split('_')[0]
        return uname

    def hyper_param(self, env, users_hyper, algorithm, epoch):
        best_discount = best_alpha = best_eps = -1
        with tqdm(total = self.prog) as pbar:
            for user in users_hyper:
                # print(user)
                max_accu = -1
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for thres in self.threshold_h:
                                env.process_data(user, thres)
                                
                                if algorithm == 'sarsa':
                                    obj = TDLearning.TDLearning()
                                    Q, stats = obj.q_learning(user, env, epoch, dis, alp, eps)    
                                else:
                                    obj = TD_SARSA.TD_SARSA()
                                    Q, stats = obj.sarsa(user, env, epoch, dis, alp, eps)

                                accu = obj.test(env, Q)
                                # print(accu)
                                if max_accu < accu:
                                    max_accu = accu
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                max_accu = max(max_accu, accu)
                                env.reset(True, False)
                                pbar.update(1)
                print("Accuracy of State Prediction: {} {} {} {} {}".format(algorithm, max_accu, best_eps, best_discount, best_alpha))
        return best_eps, best_discount, best_alpha

    def plot(self, x_labels, y, title):
        x = []
        for i in range(0, len(x_labels)):
            x.append(i)
        plt.xticks(x, x_labels)
        plt.plot(x, y)
        
        plt.xlabel('users')
        plt.ylabel('accuracy')
        plt.title(title)
        location = 'figures/' + title 
        plt.savefig(location)
        plt.close()
        # plt.show()

    def run_stuff(self, env, users, epoch, title, best_eps, best_discount, best_alpha, algo):
        x = []
        y = []
        for u in users:
            # print(u)
            sum = 0
            x.append(self.get_user_name(u))
            for episodes in tqdm(range(epoch)):
                env.process_data(u, self.threshold_h[0])
                if algo == 'sarsa':
                    # print("S")
                    obj = TDLearning.TDLearning()
                    Q, stats = obj.q_learning(u, env, 10, best_discount, best_alpha, best_eps)
                else:
                    # print("Q")
                    obj = TD_SARSA.TD_SARSA()
                    Q, stats = obj.sarsa(u, env, 10, best_discount, best_alpha, best_eps)

                accu = obj.test(env, Q)
                sum += accu
                # print("{} {}".format(u, accu))
                env.reset(True, False)
                # pdb.set_trace()
            print("{} {} {}".format(algo, u, round(sum / epoch, 2)))
            y.append(round(sum / epoch, 2))
        self.plot(x, y, title)