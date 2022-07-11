from operator import ne
from imitlearn.env_configs import *
from imitlearn.utils import printv
from imitlearn.il import get_average_reward
from rich import print

import argparse
import time
import gym
import pdb
import numpy as np
import pickle

class BlackjackQLearner:
    def __init__(self, config, exploration_rate, discount=1.0):
        self.n_actions = config['n_actions']
        self.n_attributes = config['n_attributes']

        self.exploration_rate = exploration_rate
        self.discount = discount

        self.q_values = {}
        for i in range(0, 22):
            for j in range(0, 11):
                for k in [0, 1]:
                    self.q_values[(i, j, k)] = [0, 0]
                    for a in [1, 0]:
                        if (i == 21) and (a == 0):
                            self.q_values[(i, j, k)][a] = 1
                        else:
                            self.q_values[(i, j, k)][a] = 0
    
    def act(self, state):
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return np.random.randint(0, self.n_actions)
        
        state = tuple(state)
        if np.sum(self.get_q_values(state)) == 0:
            return np.random.randint(0, self.n_actions)

        return np.argmax(self.get_q_values(state))
    
    def get_q_values(self, state):
        state = (min(state[0], 21), state[1], int(state[2]))
        return self.q_values[state]
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_values, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_values = pickle.load(f)
    
    def print(self):
        print("NO USABLE ACE")
        print(f"      {'   '.join(['{:02d}'.format(j) for j in range(1, 11)])}")
        for i in range(21, 6, -1):
            print(f" {i:02}: {['[green]H[/green]' if np.argmax(self.get_q_values((i, j, False))) else '[red]S[/red]' for j in range(1, 11)]}")
        
        print("USABLE ACE")
        print(f"      {'   '.join(['{:02d}'.format(j) for j in range(1, 11)])}")
        for i in range(21, 11, -1):
            print(f"A,{i-11}: {['[green]H[/green]' if np.argmax(self.get_q_values((i, j, True))) else '[red]S[/red]' for j in range(1, 11)]}")
    
def train_tabular_q_learner(config, args, verbose=False):
    alpha = args['learning_rate']
    gamma = args['discount']

    env = gym.make(config["name"])
    model = BlackjackQLearner(
        config, args['exploration_rate'], args['discount'])
    
    start = time.time()

    for episode in range(args['episodes']):
        state = env.reset()
        reward = 0
        done = False
        
        while not done:
            action = model.act(state)
            # if state[2]:
            #     printv(f"PLAYER: {state[0]}, DEALER: {state[1]}, USABLE ACE?: {state[2]}")
            #     printv(f"   Q_VALUES: {model.get_q_values(state)}")
            #     printv(f"   ACTION {['stick', 'hit'][action]}")

            next_state, reward, done, _ = env.step(action)
            Q1 = model.get_q_values(state)
            Q2 = model.get_q_values(next_state)

            # if done and state[2]:
            #     printv(f"   REWARD: {reward}")

            next_q = 0 if done else np.max(Q2)
            delta_q = alpha * (reward + gamma * next_q - Q1[action])
            model.q_values[state][action] += delta_q
            # if state[2]:
            #     printv(f"Added {delta_q} to action {['stick', 'hit'][action]} in state {state}")

            state = next_state
        # if state[2]:
        #     print("----")

    model.save_model(args['filepath'])
    
    env.close()
    end = time.time()

    printv(f"Time elapsed: {'{:.3f}'.format(end - start)} seconds", verbose)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Tabular Q-Learner')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-o','--filepath', help='Filepath to save model', required=True)
    parser.add_argument('-e','--episodes', help='Number of episodes to run', required=True, type=int)
    parser.add_argument('-l','--learning_rate', help='Learning rate to use (alpha)', required=True, type=float)
    parser.add_argument('-d','--discount', help='Discount factor to use (gamma)', required=True, type=float)
    parser.add_argument('--exploration_rate', help='Exploration rate to use (alpha)', required=True, type=float)
    parser.add_argument('--should_visualize', help='Should visualize final model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    if args['task'] != "blackjack":
        printv("Tabular Q-learning only available in Blackjack for now.")
    
    config = config_BJ

    model = train_tabular_q_learner(config, args, verbose=args['verbose'])
    
    print("Evaluating model...")
    model.exploration_rate = 0.0
    _, rewards = get_average_reward(config, model, episodes=10000, verbose=False)
    printv(f"Average reward for this model is {np.mean(rewards)} ± {np.std(rewards)}.")

    if args['should_visualize']:
        model.print()
    