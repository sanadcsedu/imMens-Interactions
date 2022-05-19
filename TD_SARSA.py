import pdb

import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib
import sys
import plotting
import environment2
from tqdm import tqdm

class TDlearning:
    def __init__(self):
        pass

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

        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc
    
    def sarsa(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        """
        SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
        
        Args:
            env: OpenAI environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.
        
        Returns:
            A tuple (Q, stats).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))
        
        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            # if (i_episode + 1) % 100 == 0:
            #     print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            #     sys.stdout.flush()
            
            # Reset the environment and pick the first action
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # One step in the environment
            for t in itertools.count():
                # Take a step
                next_state, reward, done, _ = env.step(state, action, True)
                
                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
        
                if done:
                    break
                    
                action = next_action
                state = next_state        
        
        return Q, stats

    def test(self, env, Q, epsilon=0.1):

        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        discount_factor = 0.1
        alpha = 0.1
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

            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # # Update statistics
            # stats.episode_rewards[i_episode] += reward
            # stats.episode_lengths[i_episode] = t
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
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

def hyper_param(users_hyper):
    discount_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # discount_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # alpha_h = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_h = [0.1, 0.2, 0.3, 0.4, 0.5]
    # # epsilon_h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilon_h = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_h = [0.75]
    # # threshold_h = [0.4, 0.6, 0.7, 0.8, 0.9]
    # # q_learning(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
    best_discount = best_alpha = best_eps = -1
    prog = 4 * len(epsilon_h) * len(alpha_h) * len(discount_h)
    with tqdm(total = prog) as pbar:
        for user in users_hyper:
            print(user)
            max_accu = -1
            for eps in epsilon_h:
                for alp in alpha_h:
                    for dis in discount_h:
                        # for thres in threshold_h:
                        env.process_data(user, 0.75)
                        obj = TDlearning()
                        Q, stats = obj.sarsa(user, env, 500, dis, alp, eps)
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
            print("Accuracy of State Prediction: {} {} {} {}".format(max_accu, best_eps, best_discount, best_alpha))


if __name__ == "__main__":
    env = environment2.environment2()
    users_b = env.user_list_bright
    users_f = env.user_list_faa
    users_hyper = []
    for i in range(2):
        c = np.random.randint(0, len(users_b))
        users_hyper.append(users_b[c])
        users_b.remove(users_b[c])

    for i in range(2):
        c = np.random.randint(0, len(users_f))
        users_hyper.append(users_f[c])
        users_f.remove(users_f[c])

    hyper_param(users_hyper)

    # thres = 0.8 #the percent of interactions Q-Learning will be trained on
    # For testing
    # for u in users_f:
    #     # print(u)
    #     sum = 0
    #     for episodes in tqdm(range(20)):
    #         env.process_data(u, thres)
    #         obj = TDlearning()
    #         Q, stats = obj.sarsa(u, env, 400, 0.1, 0.1, 0.0)
    #         # plotting.plot_episode_stats(stats)
    #         # env.take_step_subtask()
    #         # print(Q)
    #         accu = obj.test(env, Q)
    #         sum += accu
    #         # print("{} {}".format(u, accu))
    #         env.reset(True, False)
    #         # pdb.set_trace()
    #     print(sum / 20)
    
    # for u in users_b:
    #     # print(u)
    #     sum = 0
    #     for episodes in tqdm(range(20)):
    #         env.process_data(u, thres)
    #         obj = TDlearning()
    #         Q, stats = obj.sarsa(u, env, 400, 0.1, 0.1, 0.0)
    #         # plotting.plot_episode_stats(stats)
    #         # env.take_step_subtask()
    #         # print(Q)
    #         accu = obj.test(env, Q)
    #         sum += accu
    #         # print("{} {}".format(u, accu))
    #         env.reset(True, False)
    #         # pdb.set_trace()
    #     print(sum / 20)

    # print(*users_hyper, sep='\n')
    # print("xxxxxxxxxxxxxx")
    # print(*users_b, sep='\n')
    # print("xxxxxxxxxxxxxx")
    # print(*users_f, sep='\n')

    # thres = 0.9 #the percent of interactions Q-Learning will be trained on
    # for u in users_f:
    #     # print(u)
    #     env.process_data(u, thres)
    #     obj = TDlearning()
    #     Q, stats = obj.q_learning(u, env, 500)
    #     # plotting.plot_episode_stats(stats)
    #     # env.take_step_subtask()
    #     # print(Q)
    #     obj.test(env, Q)
    #     # print("OK")
    #     env.reset(True, False)