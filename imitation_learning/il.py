import gym
import random
import pdb
import time
import pickle
import numpy as np

from gc import collect
from rich import print
from collections import deque
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import imitation_learning.env_configs
from imitation_learning.utils import printv

def get_average_reward(config, model, episodes=10, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if config['should_convert_state_to_array']:
                state = np.array(state)
            
            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
        total_rewards.append(total_reward)
    
    env.close()
    
    average_reward = np.mean(total_rewards)
    printv(f"Average reward for this model is {'{:.3f}'.format(average_reward)} ± {'{:.3f}'.format(np.std(total_rewards))}.", verbose)

    return average_reward, total_rewards

def visualize_model(config, model, episodes, print_state=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        state_idx = 0
        while not done:
            state_idx += 1
            env.render()

            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            if print_state and state_idx % 5 == 0:
                for i, attribute in enumerate(config['attributes']):
                    print(f"\t'{attribute[0]}': {state[i]}")
                print(f"[red]Action[/red]: [yellow]{config['actions'][action]}[/yellow]")
                print("---")

            state = next_state
            total_reward += reward

        print(f"Episode #{episode} finished with total reward {total_reward}")
        total_rewards.append(total_reward)
    
    env.close()
    print(f"Average reward for this model is {np.mean(total_rewards)}.")

def get_dataset_from_model(config, model, episodes, verbose=False):
    env = gym.make(config["name"])
    
    X = []
    y = []

    printv("Collecting dataset from model.", verbose)
    for i in range(episodes):
        if i % 10 == 0:
            printv(f"{i} / {episodes} episodes... |D| = {len(X)}.", verbose)

        state = env.reset()
        done = False
        
        while not done:
            action = model.act(state)
            next_state, _, done, _ = env.step(action)

            X.append(state)
            y.append(action)

            state = next_state
    
    env.close()

    X = np.array(X)
    y = np.array(y)

    return X, y

def label_dataset_with_model(config, model, X):
    y = model.batch_predict(X)
    y = [np.argmax(q) for q in y]
    return y
