import pdb
import random
import sys
sys.path.insert(0, "D:\\imMens Learning\\stationarity_test")
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt


class MultiArmBandit:
    def __init__(self, vizs):
        self.vizs = vizs
        idx = 0
        self.arms = defaultdict()
        for v in vizs:
            self.arms[v] = idx
            idx += 1
        # print(self.arms)
        self.num_arms = len(vizs)
        self.counts = [0 for _ in range(len(vizs))]
        self.q_values = [0.0 for _ in range(len(vizs))]

    def reset(self, vizs):
        self.counts = [0 for _ in range(len(vizs))]
        self.q_values = [0.0 for _ in range(len(vizs))]

    def select_arm(self, epsilon):
        if random.random() > epsilon:
            return self.q_values.index(max(self.q_values))
        else:
            return random.randrange(self.num_arms)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.q_values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.q_values[chosen_arm] = new_value

    def test(self, eps, data):
        epoch = 10
        accu = 0
        for e in range(epoch):
            cnt = 0
            for idx in range(len(data)):
                arm = self.select_arm(eps)
                # pdb.set_trace()
                if self.arms[data[idx][1]] == arm:
                    cnt += 1
                self.update(arm, data[idx][2])
            accu += (cnt / len(data))
        # pdb.set_trace()
        print("eps = {} accu = {}".format(eps, round(accu / epoch, 2)))
        return round(accu / epoch, 2)

    def run_MAB(self, users_data):
        #Training the E-Greedy model from the data
        epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7]
        epsilons.sort(reverse=True)
        print(epsilons)
        idxx = 1
        for data in users_data:
            self.reset(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
            runs = 10000
            for r in range(runs):
                for idx in range(len(data)):
                    # pdb.set_trace()
                    arm = self.arms[data[idx][1]]
                    reward = data[idx][2]
                    self.update(arm, reward)
            #Testing the E-Greedy model for different Epsilon values
            accu = []
            for eps in epsilons:
                accu.append(self.test(eps, data))
            _str = "user " + str(idxx)
            idxx += 1
            plt.plot(epsilons, accu, label=_str)
        plt.ylabel('Accuracy')
        plt.xlabel('Epsilon')
        plt.legend(loc='best')
        title = 'Accuracy on different eps'
        plt.title(title)
        plt.show()


    def plot_graph(self, user):
        for v in self.cum_rewards:
            x_axis = []
            y_axis = []
            for keys in self.cum_rewards[v]:
                x_axis.append(keys[0])
                y_axis.append(keys[1])

            plt.plot(x_axis, y_axis, label=v)
            plt.ylabel('Rewards')
            # plt.xticks([])
            plt.xlabel('time')
        plt.legend(loc='best')
        title = 'Accuracy on different eps  ' + user
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    obj = integrate()
    data = obj.get_files()
    mab = MultiArmBandit(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    mab.run_MAB(data)

    # for users_data in data:
    #     mab.run_MAB(users_data)
    #     break
        # for idx in range(len(users_data)):
        #     print(users_data[idx])