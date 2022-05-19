import environment2
import numpy as np
from collections import defaultdict

class NaiveProbabilistic:
    def __init__(self):
        self.freq = defaultdict(lambda: defaultdict(float))

    def NaiveProbabilistic(self, user, env, thres): 
         
        # for t in itertools.count():
        # print(u)
        length = len(env.mem_action)
        threshold = int(length * thres)
        
        # for i in range(1, length-1):
        #     print("{} {}".format(env.mem_states[i-1], env.mem_action[i]))

        for i in range(1, threshold):
            self.freq[env.mem_states[i - 1]][env.mem_action[i]] += 1
        
        #Normalizing to get the probability 
        for states in self.freq:
            sum = 0 
            for actions in self.freq[states]:
                sum += self.freq[states][actions]
            for actions in self.freq[states]:
                self.freq[states][actions] /= sum

        #Debugging probablity calculation
        # for states in self.freq:
        #     for actions in self.freq[states]:
        #         print("{} {} {}".format(states, actions, self.freq[states][actions]))

        #Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        for i in range(threshold +1, length - 1):
            try:
                _max = max(self.freq[env.mem_states[i-1]], key = self.freq[env.mem_states[i-1]].get)
                # print(env.mem_states[i-1], _max, self.freq[env.mem_states[i-1]][_max], env.mem_action[i])
                if _max == env.mem_action[i]:
                    accuracy += 1
            except ValueError:
                pass
            denom += 1
        accuracy /= denom
        print("Accuracy {} {:.2f}".format(user, accuracy))
        self.freq.clear()
        return accuracy

if __name__ == "__main__":

    env = environment2.environment2()
    users_b = env.user_list_bright
    users_f = env.user_list_faa

    total = 0
    for u in users_b:
        env.process_data(u, 0)
        obj = NaiveProbabilistic()      
        total += obj.NaiveProbabilistic(u, env, 0.75)
        env.reset(True, False)
        # break
    for u in users_f:
        env.process_data(u, 0)
        obj = NaiveProbabilistic()
        total += obj.NaiveProbabilistic(u, env, 0.75)
        env.reset(True, False)
    
    print(total / (len(users_b) + len(users_f)))