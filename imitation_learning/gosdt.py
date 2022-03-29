import argparse
import imitation_learning.env_configs
import pdb
import time
import numpy as np

from rich import print
from imitation_learning.il import get_average_reward, get_average_reward_with_std
from imitation_learning.utils import printv, str_avg
from imitation_learning.dt_structure_viz import viztree2qtree, load_viztree
from imitation_learning.utils import load_dataset
from imodels import C45TreeClassifier

class GOSDTAgent:
    def __init__(self, model):
        self.model = model
    
    def act(self, state):
        return self.predict(state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rulelists')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-d','--dataset', help='Dataset filename', required=False, default="")
    parser.add_argument('-o','--output', help='Filepath to output converted tree', required=False)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    X, y = load_dataset(args['dataset'])
    
    model = C45TreeClassifier()
    model.fit(X, y)

    avg, std = get_average_reward_with_std(config, model, verbose=True)
    pdb.set_trace()