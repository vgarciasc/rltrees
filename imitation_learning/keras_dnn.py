from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents import DQNAgent

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import numpy as np

import keras
import gym
import pdb

class KerasDNN:
    def __init__(self, config):
        self.n_attributes = config['n_attributes']
        self.n_actions = config['n_actions']

        model = self.build_model(config)
        dqn = DQNAgent(
            model=model, 
            nb_actions=self.n_actions,
            memory=SequentialMemory(limit=50000, window_length=1), 
            nb_steps_warmup=10,
            target_model_update=1e-2, 
            policy=BoltzmannQPolicy())
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        
        self.dqn = dqn
    
    def predict(self, s):
        s = np.reshape(s, (1, self.n_attributes))
        return self.dqn.compute_q_values(s)
    
    def batch_predict(self, X):
        X = np.reshape(X, (len(X), 1, self.n_attributes))
        return self.dqn.compute_batch_q_values(X)
    
    def act(self, state):
        q_values = self.predict(state)
        return np.argmax(q_values)

    def build_model(self, env):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.n_attributes,)))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(self.n_actions))
        model.add(Activation('linear'))
        return model
    
    def save(self, filename):
        self.dqn.model.save(filename)
    
    def load(self, filename):
        self.dqn.model = keras.models.load_model(filename)

if __name__ == '__main__':
    mc_config = {
        "name": "MountainCar-v0",
        "n_actions": 3,
        "n_attributes": 2
    }

    ll_config = {
        "name": "LunarLander-v2",
        "n_actions": 4,
        "n_attributes": 8
    }

    config = ll_config

    env = gym.make(config['name'])
    ann = KerasDNN(config)

    ann.dqn.fit(env, nb_steps=150000, visualize=False, verbose=2)
    ann.dqn.test(env, nb_episodes=25, visualize=True)
    ann.save(f"data/{config['name']}_keras-nn")

    # ann.load(f"data/{config['name']}_keras-nn")
    # ann.dqn.test(env, nb_episodes=25, visualize=True)