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

GAMMA = 0.95
ALPHA = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

class MLPAgent:
    def __init__(self, config, exploration_rate=EXPLORATION_MAX):
        self.n_attributes = config["n_attributes"]
        self.n_actions = config["n_actions"]

        self.exploration_rate = exploration_rate
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.n_attributes,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.n_actions, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=ALPHA))
    
    def predict(self, state):
        s = np.reshape(state, (1, self.n_attributes))
        return self.model.predict(s)[0]
    
    def fit(self, state, target):
        s = np.reshape(state, (1, self.n_attributes))
        t = np.reshape(target, (1, self.n_actions))
        self.model.fit(s, t, verbose=0)

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
        
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            if done:
                target_q = reward
            else:
                target_q = reward + GAMMA * np.amax(self.predict(next_state))
            
            target = self.predict(state)
            target[action] = target_q

            self.fit(state, target)
        
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = keras.models.load_model(filename)

if __name__ == "__main__":
    config = {
        "name": "CartPole-v1",
        "can_render": True,
        "episode_max_score": 195,
        "should_force_episode_termination_score": True,
        "episode_termination_score": 0,
        "n_actions": 2,
        "actions": ["left", "right"],
        "n_attributes": 4,              
        "attributes": [
            ("Cart Position", "continuous", -1, -1),
            ("Cart Velocity", "continuous", -1, -1),
            ("Pole Angle", "continuous", -1, -1),
            ("Pole Angular Velocity", "continuous", -1, -1)],
    }

    collect_data(config)

    # model = load_model("data/cartpole_nn_2")
    # get_average_reward(model, verbose=True)

    # model = load_model(config, "data/cartpole_nn_7")
    # visualize_model(config, model)
