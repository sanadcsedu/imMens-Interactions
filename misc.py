# contains all the miscellaneous functions for running
import pdb
import random

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
from random import randint


# import numba

class misc:
    def __init__(self, users):
        self.discount_h = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.alpha_h = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.epsilon_h = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.threshold_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.prog = 2 * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h) * len(self.threshold_h)
        self.store_acc = []
        self.store_threshold = []
        self.store_uname = []

    def get_user_name(self, url):
        string = url.split('\\')
        fname = string[len(string) - 1]
        uname = fname.split('-')[0]
        return uname

    def plot_together(self, store_acc, store_uname, algorithm):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(store_acc)):
            plt.plot(self.threshold_h, store_acc[i], label=store_uname[i])

        for i in range(len(store_acc)):
            for x, y in zip(self.threshold_h, store_acc[i]):
                label = "{:.2f}".format(y)
                plt.annotate(label,  # this is the text
                             (x, y),  # these are the coordinates to position the label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 0),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        title = algorithm + "_" + str(randint(100, 999))
        # pdb.set_trace()
        plt.title(title)
        location = 'figures/' + title
        plt.savefig(location, bbox_inches='tight')
        plt.close()

    def hyper_param(self, env, users_hyper, algorithm, epoch):
        # print(users_hyper)
        best_discount = best_alpha = best_eps = -1
        e = a = d = 0
        pp = 10  # number of time you should take the average
        with tqdm(total=self.prog) as pbar:
            for user in users_hyper:
                # print(user)
                max_accu = -1
                # x_thres = []
                y_accu = []
                for thres in self.threshold_h:
                    max_accu_thres = -1
                    env.process_data(user, thres)
                    for eps in self.epsilon_h:
                        for alp in self.alpha_h:
                            for dis in self.discount_h:
                                accu = 0
                                for epiepi in range(pp):
                                    if algorithm == 'qlearning':
                                        obj = TDLearning.TDLearning()
                                        Q = obj.q_learning(user, env, epoch, dis, alp, eps)
                                    else:
                                        obj = TD_SARSA.TD_SARSA()
                                        Q = obj.sarsa(user, env, epoch, dis, alp, eps)
                                    # pdb.set_trace()
                                    accu += obj.test(env, Q, dis, alp, eps)

                                if max_accu_thres < accu:
                                    max_accu = accu
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                max_accu_thres = max(max_accu_thres, accu)

                                pbar.update(1)
                    env.reset(True, False)
                    y_accu.append(round(max_accu_thres / pp, 2))
                    max_accu = max(max_accu_thres, max_accu)
                    # print(user, thres, best_eps, round(max_accu_thres / pp, 2))
                    # pdb.set_trace()
                # print(self.threshold_h, y_accu, self.get_user_name(user))
                # plt.plot(self.threshold_h, y_accu, label = self.get_user_name(user))
                # pdb.set_trace()
                self.store_acc.append(y_accu)
                self.store_uname.append(self.get_user_name(user))
                # print(self.store_acc, self.store_uname)
                # print("{}, {:.2f}, {}, {}, {}".format(self.get_user_name(user), max_accu/pp, best_eps, best_discount, best_alpha))
                e += best_eps
                d += best_discount
                a += best_alpha
        # return best_eps, best_discount, best_alpha
        # print(e / len(users_hyper), d / len(users_hyper), a / len(users_hyper))
        return e / len(users_hyper), d / len(users_hyper), a / len(users_hyper), self.store_acc, self.store_uname

    def plot(self, x_labels, y, title):
        x = []
        for i in range(0, len(x_labels)):
            x.append(i)
        plt.xticks(x, x_labels)
        plt.plot(x, y)

        plt.xlabel('Users')
        plt.ylabel('Accuracy')
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
                    Q, stats = obj.q_learning(u, env, 50, best_discount, best_alpha, best_eps)
                else:
                    # print("Q")
                    obj = TD_SARSA.TD_SARSA()
                    Q = obj.sarsa(u, env, 50, best_discount, best_alpha, best_eps)

                accu = obj.test(env, Q, best_discount, best_alpha, best_eps)
                sum += accu
                # print("{} {}".format(u, accu))
                env.reset(True, False)
                # pdb.set_trace()
            print("{} {} {}".format(algo, u, round(sum / epoch, 2)))
            y.append(round(sum / epoch, 2))
        self.plot(x, y, title)