import pdb
import random
import sys
sys.path.insert(0, "/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test")
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import scipy.stats as st

class mortal_bandit:

    def __init__(self):
        pass
    def stochastic_early_stop(self, data, threshold):
        viz = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
        mean = dict()
        sz = len(data)
        e_idx = int(threshold * sz)
        # print(e_idx, threshold, len(data))
        # pdb.set_trace()
        ret = 0
        limit = len(data) - e_idx
        for L in range(1, min(15, limit)):
            # print(L)
            argmax_mean = -1
            best_arm = None
            for v in viz:
                #If we train this reward distribution from the data
                param, dist = self.get_distribution(data, v, e_idx)
                alpha = param[0]
                beta = param[1]
                loc = param[2]
                scale = param[3]
                # X = dist.ppf(np.random.rand(), alpha, beta, loc, scale)

                #if we assume that our reward comes from a beta distribution
                # alpha = 2
                # beta = 3
                # loc = 0
                # scale = 1
                # dist = st.beta.rvs(alpha, beta, 1000)
                E_X = alpha / (alpha + beta)
                mean[v] = E_X
                F_mean = dist.cdf(E_X, alpha, beta, loc, scale) #CDF at mean
                # F_mean = st.beta.cdf(E_X, alpha, beta)
                E_X_givenXgreaterE = 2 * E_X / (1 - F_mean)
                up = E_X + (1 - F_mean) * (L - 1) * E_X_givenXgreaterE
                down = (1 + (1 - F_mean) * (L - 1))
                gamma = up / down
                if argmax_mean < gamma:
                    argmax_mean = gamma
                    best_arm = viz
                # pdb.set_trace()
            # print(argmax_mean)
            # print(mean)
            best_accu = 0
            for n in range(1, min(L, limit)):
                avg_accu = 0
                for runs in range(10):
                    cnt = 0
                    denom = 0
                    idx = e_idx
                    while idx < len(data):
                        i = random.choice(viz)
                        # print("Initial Random Pulling {} {}".format(data[idx][1], i))
                        # pdb.set_trace()
                        r = 0
                        d = 0
                        l = 0
                        flag = 0
                        while d < n and n - d >= n * argmax_mean - r and idx < len(data) and l < L:
                            #Keep pulling arm i
                            if data[idx][1] == i:
                                cnt += 1
                                r += data[idx][2]
                            else:
                                r += 0
                            idx += 1
                            denom += 1
                            l += 1
                            # pdb.set_trace()
                            # if random.random() < mean[i]:
                            #     r = r + 1
                            # else:
                            #     r = r + 0

                            # r += mean[i]
                            d = d + 1
                            # pdb.set_trace()
                            flag = 1
                        if flag == 0:
                            if data[idx][1] == i:
                                cnt += 1
                            idx += 1
                            denom += 1
                        if r > n * argmax_mean:
                            #pull arm i untill lifetime
                            while l < L and idx < len(data):
                                # pdb.set_trace()
                                l += 1
                                # print("Repeated Pulling {} {}".format(data[idx][1], i))
                                if data[idx][1] == i:
                                    cnt += 1
                                idx += 1
                                denom += 1
                        # pdb.set_trace()
                    avg_accu += cnt / denom
                # print("L {} n = {} Accu {}".format(L, n, round(avg_accu / 10, 2)))
                best_accu = max(best_accu, avg_accu / 10)
            ret = max(ret, best_accu)
        return round(ret, 2)

    def stochastic_early_stop_regret(self, data, runs):
        viz = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
        mean = dict()
        sz = len(data)
        ret = 0
        best_reward = -1
        for L in range(1, min(20, sz)):
            argmax_mean = -1
            for v in viz:
                #If we train this reward distribution from the data
                param, dist = self.get_distribution(data, v, sz)
                alpha = param[0]
                beta = param[1]
                loc = param[2]
                scale = param[3]
                E_X = alpha / (alpha + beta)
                mean[v] = E_X
                F_mean = dist.cdf(E_X, alpha, beta, loc, scale) #CDF at mean
                E_X_givenXgreaterE = 2 * E_X / (1 - F_mean)
                up = E_X + (1 - F_mean) * (L - 1) * E_X_givenXgreaterE
                down = (1 + (1 - F_mean) * (L - 1))
                gamma = up / down
                if argmax_mean < gamma:
                    argmax_mean = gamma

            for n in range(1, L):
                model_reward = 0
                for z in range(5):
                    idx = 0
                    while idx < len(data):
                        i = random.choice(viz)
                        r = 0
                        d = 0
                        l = 0
                        flag = 0
                        while d < n and n - d >= n * argmax_mean - r and idx < len(data) and l < L:
                            #Keep pulling arm i
                            if data[idx][1] == i:
                                model_reward += data[idx][2]
                                r += data[idx][2]
                            else:
                                r += 0
                            idx += 1
                            l += 1
                            # pdb.set_trace()
                            # if random.random() < mean[i]:
                            #     r = r + 1
                            # else:
                            #     r = r + 0

                            # r += mean[i]
                            d = d + 1
                            # pdb.set_trace()
                            flag = 1
                        if flag == 0:
                            if data[idx][1] == i:
                                model_reward += data[idx][2]
                            idx += 1
                        if r > n * argmax_mean:
                            while l < L and idx < len(data):
                                l += 1
                                if data[idx][1] == i:
                                    model_reward += data[idx][2]
                                idx += 1
                    # pdb.set_trace()
                best_reward = max(best_reward, model_reward / 5)
                # print(best_reward)
            ret = max(ret, best_reward)
        return round(ret, 2)
    def get_best_distribution(self, data):
        # dist_names = ['gamma', 'rayleigh', 'norm', 'pareto', 'weibull_min', 'weibull_max','beta',
        #       'invgauss','uniform', 'expon', 'lognorm', 'pearson3','triang']
        dist_names = ['beta']
        dist_results = []
        params = {}
        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            param = dist.fit(data)
            # pdb.set_trace()
            params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = st.kstest(data, dist_name, args=param)
            # print("p value for " + dist_name + " = " + str(p))
            dist_results.append((dist_name, p))

        # select the best fitted distribution
        best_dist_name, best_p = (max(dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value

        # print("Best fitting distribution: " + str(best_dist_name))
        # print("Best p value: " + str(best_p))
        # print("Parameters for the best fit: " + str(params[best_dist_name]))

        # return best_dist, best_p, params[best_dist]
        ret_best_dist = getattr(st, best_dist_name)
        # param = ret_best_dist.fit(data)
        return ret_best_dist

    def get_distribution(self, data, viz, e_idx):
        ret = np.zeros(e_idx)
        idx = 0
        for idx in range(e_idx - 1):
            # pdb.set_trace()
            if viz != data[idx][1]:
                continue
            ret[idx] = data[idx][2]
            # ret[idx] = 1
        ret[idx] = 1
        dist_best_fit = self.get_best_distribution(ret)
        param = dist_best_fit.fit(ret)
        return param, dist_best_fit

    def run_stochastic_early_stop(self, threshold, data):
        accu_list = []
        for t in threshold:
            accu_list.append(self.stochastic_early_stop(data, t))
        return accu_list


if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    mortal = mortal_bandit()
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # pdb.set_trace()
    # threshold = [0.5]
    idx = 0
    for users_data in data:
        accu_list = []
        for t in threshold:
            accu_list.append(mortal.stochastic_early_stop(users_data, t))
            # pdb.set_trace()
        print(uname[idx], accu_list)
        idx += 1
        # break