from tabnanny import verbose
import gym
# import gym_snake
import random
import pdb
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gc import collect
from rich import print
from collections import deque
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1
from keras.optimizers import Adam

import imitlearn.env_configs 
from imitlearn.distilled_tree import DistilledTree
from imitlearn.utils import printv
from imitlearn.il import *

GAMMA = 0.99
ALPHA = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9999

MEMORY_SIZE = 500000
BATCH_SIZE = 10

class MLPAgent:
    def __init__(self, config, exploration_rate=EXPLORATION_MAX):
        self.n_attributes = config["n_attributes"]
        self.n_actions = config["n_actions"]
        self.config = config

        self.exploration_rate = exploration_rate
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(16, input_shape=(self.n_attributes,), activation="relu", kernel_regularizer=l1(0.01)))
        self.model.add(Dense(16, activation="relu", kernel_regularizer=l1(0.01)))
        self.model.add(Dense(self.n_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))
    
    def predict(self, state):
        s = np.reshape(state, (1, self.n_attributes))
        return self.model.predict(s, verbose=0)[0]
    
    def batch_predict(self, X):
        X = np.reshape(X, (len(X), self.n_attributes))
        return self.model.predict(X)

    def fit(self, state, target):
        s = np.reshape(state, (1, self.n_attributes))
        t = np.reshape(target, (1, self.n_actions))
        self.model.fit(s, t, verbose=0)

    def batch_fit(self, X, y):
        self.model.fit(X, y, verbose=0)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.n_actions)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        X = []
        y = []

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in batch:
            if done:
                target_q = reward
            else:
                target_q = reward + GAMMA * np.amax(self.predict(next_state))
            
            target = self.predict(state)
            target[action] = target_q
            
            X.append(state)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # start_time = time.time()
        # dt = DistilledTree(self.config)
        # dt.fit()
        # print(f"Training the DT with {dt.get_size()} nodes took {time.time() - start_time} seconds.")
        
        self.batch_fit(X, y)
        
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = keras.models.load_model(filename)

def collect_data(config, args, verbose=False):
    env = gym.make(config["name"])
    model = MLPAgent(config)

    best_reward = -9999
    best_model = None
    
    start = time.time()
    history = []

    for episode in range(args['episodes']):
        raw_state = env.reset()
        state = config['conversion_fn'](env, None, raw_state)
        reward = 0
        done = False
        
        while not done:
            action = model.act(state)

            raw_next_state, reward, done, _ = env.step(action)
            next_state = config['conversion_fn'](env, raw_state, raw_next_state)
            if config['should_force_episode_termination_score']:
                reward = reward if not done else config['episode_termination_score']

            model.remember(state, action, reward, next_state, done)
            model.experience_replay()

            state = next_state
            raw_state = raw_next_state

        avg_reward, rewards = get_average_reward(config, model, episodes=20)
        deviation = np.std(rewards)
        printv(f"Model at episode #{episode} has average reward {avg_reward} ± {deviation}", verbose)
        history.append((episode, avg_reward, deviation))

        if avg_reward > best_reward:
            printv(f"> Saving new best model...", verbose)
            best_reward = avg_reward
            best_model = model
            model.save_model(args['filepath'])
    
    env.close()
    end = time.time()

    printv(f"Time elapsed: {'{:.3f}'.format(end - start)} seconds", verbose)

    return best_model, best_reward, zip(*history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ANN')
    parser.add_argument('-t','--task', help="Which task to run?", required=True)
    parser.add_argument('-o','--filepath', help='Filepath to save model', required=True)
    parser.add_argument('-e','--episodes', help='Number of episodes to run', required=True, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_plot', help='Should plot training performance?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_visualize', help='Should visualize final model?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    # Initialization
    config = imitlearn.env_configs.get_config(args['task'])

    # Running training for NN
    _, reward, history = collect_data(config, args, args['verbose'])

    # Getting expert
    expert = MLPAgent(config, exploration_rate=0.0)
    expert.load_model(args['filepath'])
    
    # Grading NN
    get_average_reward(config, expert, episodes=10, verbose=True)

    # Plotting
    if args['should_plot']:
        iterations, avg_rewards, deviations = history
        
        avg_rewards = np.array(avg_rewards)
        deviations = np.array(deviations)

        plt.fill_between(iterations, avg_rewards - deviations, avg_rewards + deviations,
            color="red", alpha=0.2)
        plt.plot(iterations, avg_rewards, color="red")
        plt.ylabel("Average reward")
        plt.xlabel("Iterations")
        plt.title(f"Training performance for NN in {config['name']}")
        plt.show()

    if args['should_visualize']:
        visualize_model(config, expert, 25)
