import pdb
import numpy as np
import matplotlib.pyplot as plt

import ann
from il import *
from imitation_learning.utils import load_dataset, printv, save_dataset
from imitation_learning.keras_dnn import KerasDNN

def handle_args(args, config):
    filename = args['expert_filepath']

    if args['expert_class'] == "KerasDNN":
        expert = KerasDNN(config)
        expert.load(filename)
    elif args['expert_class'] == "MLP":
        expert = ann.MLPAgent(config, exploration_rate=0)
        expert.load_model(filename)

    if args['should_grade_expert']:
        avg_reward, rewards = get_average_reward(config, expert, verbose=True)
        printv(f"Average reward for the expert: {avg_reward} ± {np.std(rewards)}.")
        printv("")

    if args['should_collect_dataset']:
        X, y = get_dataset_from_model(config, expert, args['dataset_size'])
        save_dataset(f"{filename}_dataset", X, y)
    else:
        X, y = load_dataset(f"{filename}_dataset")
        
    return expert, X, y