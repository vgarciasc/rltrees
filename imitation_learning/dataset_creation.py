import argparse
import imitation_learning.env_configs
import pdb

from imitation_learning.utils import save_dataset
from il import get_dataset_from_model

from functools import reduce
from collections import Counter
from rich import print
from imitation_learning.il import get_average_reward, get_average_reward_with_std, label_dataset_with_model
from qtree import QNode, QLeaf, load_tree
from statsmodels.stats.proportion import proportion_confint
from imitation_learning.utils import printv, str_avg
from imitation_learning.dt_structure_viz import viztree2qtree, load_viztree
from imitation_learning.utils import load_dataset
from rulelists import Rulelist
from ann import MLPAgent
from keras_dnn import KerasDNN

def get_model(model_class, filename, config):
    if model_class == "QTree":
        model = load_tree(filename)
    elif model_class == "VizTree":
        string = load_viztree(filename)
        model = viztree2qtree(config, string)
    elif model_class == "Rulelist":
        model = Rulelist(config)
        model.load_txt(filename)
    elif model_class == "MLP":
        model = MLPAgent(config, 0.0)
        model.load_model(filename)
    elif model_class == "KerasDNN":
        model = KerasDNN(config, 0.0)
        model.load_model(filename)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rulelists')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-s','--student', help='Filepath for student', required=True)
    parser.add_argument('--student_class', help='Which type of file for the student?', required=True)
    parser.add_argument('-e','--expert', help='Filepath for expert', required=False)
    parser.add_argument('--expert_class', help='Which type of file for the expert?', required=False)
    parser.add_argument('-o','--output', help='Filepath to output converted tree', required=False)
    parser.add_argument('--dataset_size', help='Number of episodes used during pruning.', required=False, default=10, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    config = imitation_learning.env_configs.get_config(args['task'])
    print(f"[yellow]Creating dataset for {config['name']} with {args['dataset_size']} episodes.")
    print("")

    student = get_model(
        model_class=args['student_class'],
        filename=args['student'],
        config=config)
    
    X, y = get_dataset_from_model(
        config, student, 
        episodes=args['dataset_size'],
        verbose=args['verbose'])
    
    if args['expert']:
        printv(f"[yellow]Labeling dataset with expert '{args['expert']}'...[/yellow]")
        expert = get_model(
            model_class=args['expert_class'],
            filename=args['expert'],
            config=config)
        y = label_dataset_with_model(
            config=config,
            model=expert,
            X=X)
    
    save_dataset(args['output'], X, y)
    printv(f"[yellow]Saved dataset to '{args['output']}'.[/yellow]")