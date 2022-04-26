import argparse
import gym
import pdb
import time
import numpy as np
from dataset_creation import get_model

from rulelists import Rulelist
from qtree import load_tree, QLeaf, QNode
from distilled_tree import DistilledTree
from il import visualize_model
from il import get_average_reward
from qtree import save_tree_from_print
from rich import print
from dt_structure_viz import viztree2qtree, load_viztree
from utils import load_dataset

import imitation_learning.env_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checking model accuracy')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for model', required=True)
    parser.add_argument('-c','--class', help='Tree is QTree, Distilled Tree, or Viztree?', required=True)
    parser.add_argument('-d','--dataset', help='Dataset to check accuracy for', required=True)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    model = get_model(args['class'], args['filename'], config)

    start_time = time.time()
    
    X, y = load_dataset(args['dataset'])
    y_pred = [model.act(x) for x in X]
    correct_preds = sum([(1 if y[i] == y_pred[i] else 0) for i in range(len(X))])
    accuracy_insample = correct_preds / len(X)

    print(f"Accuracy in-sample: {'{:.3f}'.format(accuracy_insample * 100)}%")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds.")