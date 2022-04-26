import argparse
import imitation_learning.env_configs
import pdb
import time
import pickle
import numpy as np
import gosdt
import pandas as pd

from rich import print
from il import get_average_reward, get_average_reward_with_std
from utils import printv, str_avg
from dt_structure_viz import viztree2qtree, load_viztree
from utils import load_dataset

class GOSDTAgent:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y, pruning=0):
        print("Now fitting...")

        df = pd.DataFrame(np.hstack((X, y.reshape(len(y), 1))))
        X = df[df.columns[:-1]]
        y = df[df.columns[-1:]]
        
        pdb.set_trace()
        clf = gosdt.GOSDT()
        clf = clf.fit(X, y)
        self.model = clf

    def act(self, state):
        state = state.reshape(1, -1)
        action = self.model.predict(state)
        action = action[0]
        return action

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved tree to '{filename}'.")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        
    def get_size(self):
        return self.model.get_n_leaves() * 2 - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GOSDT')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-d','--dataset', help='Dataset filename', required=False, default="")
    parser.add_argument('-o','--output', help='Filepath to output converted tree', required=False)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    X, y = load_dataset(args['dataset'])
    
    model = GOSDTAgent(config)
    model.fit(X, y)

    avg, std = get_average_reward_with_std(config, model, verbose=True)
    pdb.set_trace()