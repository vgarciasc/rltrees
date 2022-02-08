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
from il import *

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

        self.exploration_rate = exploration_rate
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(self.n_attributes,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(self.n_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))
    
    def predict(self, state):
        s = np.reshape(state, (1, self.n_attributes))
        return self.model.predict(s)[0]
    
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
        
        self.batch_fit(X, y)
        
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = keras.models.load_model(filename)

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
        
        while not done:
            action = model.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -1

            model.remember(state, action, reward, next_state, done)
            model.experience_replay()

            state = next_state

        avg_reward, rewards = get_average_reward(config, model, episodes=20)
        deviation = np.std(rewards)
        print(f"Model at episode #{episode} has average reward {avg_reward} ± {deviation}")
        total_rewards.append(avg_reward)

        if avg_reward > best_reward:
            print(f"> Saving new best model...")
            best_reward = avg_reward
            model.save_model(f"data/mountaincar_nn_{model_id}")
            model_id += 1
    
    env.close()
    end = time.time()

    print(f"Average reward per episode: {np.mean(total_rewards)} ± {np.std(total_rewards)}")
    print(f"Time elapsed: {'{:.3f}'.format(end - start)} seconds")

if __name__ == "__main__":
    # config = {
    #     "name": "CartPole-v1",
    #     "can_render": True,
    #     "episode_max_score": 195,
    #     "should_force_episode_termination_score": True,
    #     "episode_termination_score": 0,
    #     "n_actions": 2,
    #     "actions": ["left", "right"],
    #     "n_attributes": 4,              
    #     "attributes": [
    #         ("Cart Position", "continuous", -1, -1),
    #         ("Cart Velocity", "continuous", -1, -1),
    #         ("Pole Angle", "continuous", -1, -1),
    #         ("Pole Angular Velocity", "continuous", -1, -1)],
    # }

    config = {
        "name": "LunarLander-v2",
        "can_render": True,
        "n_actions": 4,
        "actions": ["nop", "left engine", "main engine", "right engine"],
        "n_attributes": 8,              
        "attributes": [
            ("X Position", "continuous", -1, -1),
            ("Y Position", "continuous", -1, -1),
            ("X Velocity", "continuous", -1, -1),
            ("Y Velocity", "continuous", -1, -1),
            ("Angle", "continuous", -1, -1),
            ("Angular Velocity", "continuous", -1, -1),
            ("Leg 1 is Touching", "binary", [0, 1], -1),
            ("Leg 2 is Touching", "binary", [0, 1], -1)],
    }

    # config = {
    #     "name": "MountainCar-v0",
    #     "can_render": True,
    #     "episode_max_score": 195,
    #     "should_force_episode_termination_score": False,
    #     "episode_termination_score": 0,
    #     "should_stop_if_no_splits": False,
    #     "max_iters_without_split": 3,
    #     "n_actions": 2,
    #     "actions": ["left", "right"],
    #     "n_attributes": 2,              
    #     "attributes": [("Car Position", "continuous", -1, -1),
    #                     ("Car Velocity", "continuous", -1, -1)],
    # }

    collect_data(config)

    # model = MLPAgent(config, exploration_rate=0.0)
    # model.load_model("data/lunarlander_nn_8")
    # get_average_reward(config, model, verbose=True)
    # visualize_model(config, model, 10)
