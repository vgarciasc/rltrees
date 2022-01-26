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

from imitation_learning.utils import printv

def collect_data(config):
    env = gym.make(config["name"])
    model = MLPAgent(config)

    best_reward = 0
    best_model = None
    model_id = 1
    total_rewards = []
    
    start = time.time()

    for episode in range(300):
        state = env.reset()
        reward = 0
        done = False
        total_reward = 0
        
        while not done:
            action = model.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -1

            model.remember(state, action, reward, next_state, done)
            model.experience_replay()

            state = next_state
            total_reward += reward

        print(f"Episode #{episode} finished with total reward {total_reward}")
        total_rewards.append(total_reward)

        if total_reward > best_reward:
            print(f"> Saving new best model...")
            best_reward = total_reward
            model.save_model(f"data/cartpole_nn_{model_id}")
            model_id += 1
    
    env.close()
    end = time.time()

    print("Average reward per episode:", np.mean(total_rewards))
    print(f"Time elapsed: {'{:.3f}'.format(end - start)} seconds")

def get_average_reward(config, model, episodes=10, verbose=False):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        printv(f"Episode #{episode} finished with total reward {total_reward}", verbose)
        total_rewards.append(total_reward)
    
    env.close()
    
    average_reward = np.mean(total_rewards)
    printv(f"Average reward for this model is {average_reward}.", verbose)

    return average_reward

def visualize_model(config, model, episodes):
    env = gym.make(config["name"])
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            env.render()

            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        print(f"Episode #{episode} finished with total reward {total_reward}")
        total_rewards.append(total_reward)
    
    env.close()
    print(f"Average reward for this model is {np.mean(total_rewards)}.")

def load_and_visualize(config, filename):
    env = gym.make(config["name"])
    model = MLPAgent(env, exploration_rate=0)
    model.load_model(filename)

    total_rewards = []

    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            env.render()

            action = model.act(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        print(f"Episode #{episode} finished with total reward {total_reward}")
        total_rewards.append(total_reward)
    
    env.close()
    print(f"Average reward for this model is {np.mean(total_rewards)}.")