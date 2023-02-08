import pdb
import random
import sys
sys.path.insert(0, "D:\\imMens Learning\\stationarity_test")
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import scipy.stats as st

class mortal_bandit:

    def stochastic_early_stop(self, data, threshold):
        viz = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
        mean = dict()
        e_idx = int(threshold * len(data))
        ret = 0
        for L in range(1, 40):
            # print(L)
            argmax_mean = -1
            best_arm = None
            for v in viz:
                param, dist = self.get_distribution(data, v, e_idx)
                # pdb.set_trace()
                alpha = param[0]
                beta = param[1]
                loc = param[2]
                scale = param[3]
                # X = dist.ppf(np.random.rand(), alpha, beta, loc, scale)
                E_X = alpha / (alpha + beta)
                mean[v] = E_X
                F_mean = dist.cdf(E_X, alpha, beta, loc, scale) #CDF at mean
                E_X_givenXgreaterE = 2 * E_X / (1 - F_mean)
                gamma = E_X + (1 - F_mean) * (L - 1) * E_X_givenXgreaterE
                gamma = gamma / (1 + (1 - F_mean) * (L - 1))
                if argmax_mean < gamma:
                    argmax_mean = gamma
                    best_arm = viz
            # print(argmax_mean)
            # print(mean)
            best_accu = 0
            for n in range(1, int((len(data) - e_idx) / 2)):
                avg_accu = 0
                for runs in range(10):
                    cnt = 0
                    denom = 0
                    idx = e_idx
                    while idx < len(data):
                        denom += 1
                        i = random.choice(viz)
                        if data[idx][1] == i:
                            cnt += 1
                        idx += 1
                        r = 0
                        d = 0
                        while d < n and n - d > n * argmax_mean - r:
                            #Keep pulling arm i
                            if random.random() < mean[i]:
                                r = r + 1
                            else:
                                r = r + 0
                            # r += mean[i]
                            d = d + 1
                        l = 0
                        if r > n * argmax_mean:
                            #pull arm i untill lifetime
                            while l < L and idx < len(data):
                                # print("Here")
                                if data[idx][1] == i:
                                    cnt += 1
                                idx += 1
                                denom += 1
                    avg_accu += cnt / denom
                best_accu = max(best_accu, avg_accu / 10)
            ret = max(ret, best_accu)
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
        for idx in range(e_idx):
            # pdb.set_trace()
            if viz != data[idx][1]:
                continue
            ret[idx] = data[idx][2]
        # print(ret)
        dist_best_fit = self.get_best_distribution(ret)
        param = dist_best_fit.fit(ret)
        return param, dist_best_fit

if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    mortal = mortal_bandit()
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # threshold = [0.5]
    idx = 0
    for users_data in data:
        accu_list = []
        for t in threshold:
            accu_list.append(mortal.stochastic_early_stop(users_data, t))
        print(uname[idx], accu_list)
        idx += 1
        # break