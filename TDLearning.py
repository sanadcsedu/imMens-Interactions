import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import sys
import plotting
import environment5
from tqdm import tqdm
# from numba import jit, cuda 
import multiprocessing
import time
from multiprocessing import Pool


class TDLearning:
    def __init__(self):
        pass

    # @jit(target ="cuda")
    def epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """
        
        # @jit(target ="cuda")
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    # @jit(target ="cuda")
    def q_learning(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: setting the environment as local fnc by importing env earlier
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))

        # Keeps track of useful statistics
        # stats = plotting.EpisodeStats(
        #     episode_lengths=np.zeros(num_episodes),
        #     episode_rewards=np.zeros(num_episodes))
        # The policy we're following
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

        # for i_episode in tqdm(range(num_episodes)):
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            # if (i_episode + 1) % 100 == 0:
            #     print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            #     sys.stdout.flush()

            # Reset the environment and pick the first state
            state = env.reset()

            # One step in the environment
            # total_reward = 0.0
            # print("episode")
            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(state, action, False)
                # pdb.set_trace()
                # Update statistics
                # stats.episode_rewards[i_episode] += reward
                # stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                # pdb.set_trace()
                if done:
                    break
                state = next_state
        # print(policy)
        return Q

    # @jit(target ="cuda")
    def test(self, env, Q, discount_factor, alpha, epsilon):

        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        # Reset the environment and pick the first action
        state = env.reset(all = False, test=True)

        stats = []
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, prediction = env.step(state, action, True)

            stats.append(prediction)
            # print(prediction)
            # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions 
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

        cnt = 0
        for i in stats:
            cnt += i
        cnt /= len(stats)
        # print("Accuracy of State Prediction: {}".format(cnt))
        return cnt


if __name__ == "__main__":
    start_time = time.time()
    env = environment5.environment5()
    # users_b = env.user_list_bright
    users_f = env.user_list_faa
    # pdb.set_trace()
    obj2 = misc.misc(len(users_f))
    episodes = 50
    a, b, c, d, e = obj2.hyper_param(env, users_f, 'qlearning', episodes)
    pdb.set_trace()
    # with Pool(4) as P:
    #     results = P.starmap(obj2.hyper_param, [(env, users_f[0:2], 'qlearning', episodes),
    #                                            (env, users_f[2:4], 'qlearning', episodes),
    #                                            (env, users_f[4:6], 'qlearning', episodes),
    #                                            (env, users_f[6:8], 'qlearning', episodes)])
    # accu_list = []
    # name_list = []
    # for i in range(len(results)):
    #     for items in results[i][3]:
    #         accu_list.append(items)
    #     for names in results[i][4]:
    #         name_list.append(names)
    #
    # obj2.plot_together(accu_list, name_list, 'qlearning')

#Threshold =  [[0.10, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.80, 0.90]]
#Q-Learning = [[0.57, 0.52, 0.56, 0.63, 0.6, 0.6, 0.61, 0.61, 0.65]]
#Baseline =  [[0.89, 0.88, 0.88, 0.87, 0.84, 0.87, 0.92, 0.94, 0.94]]