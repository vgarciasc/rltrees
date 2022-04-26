import pdb
import numpy as np
import argparse
import imitation_learning.env_configs
from utils import load_dataset, str_avg
from dataset_creation import get_model
from il import get_average_reward, get_average_reward_with_std

from rulelists import UCF
from rich import print
from qtree import QNode, QLeaf, load_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decision Tree Neural Network Grafting')
    parser.add_argument('-t','--task',help="Which task?", required=True)
    parser.add_argument('-f','--filepath',help="Which tree to view?", required=True)
    parser.add_argument('-c','--class',help="Which model class to use?", required=True)
    parser.add_argument('-d','--dataset',help="Which dataset to use?", required=True)
    parser.add_argument('-e','--expert', help='Which expert to load, if needed', required=True)
    parser.add_argument('-x','--expert_class', help='What is the expert\'s class', required=True)
    parser.add_argument('--grading_episodes', help='How many episodes should we use to measure model\'s accuracy?', required=False, default=100, type=int)
    args = vars(parser.parse_args())

    config = imitation_learning.env_configs.get_config(args['task'])
    tree = get_model(args['class'], args['filepath'], config)
    expert = get_model(args['expert_class'], args['expert'], config)

    X, y = load_dataset(args['dataset'])
    dataset_size = len(X)
    leaf_samples_pairs = tree.get_covered_samples(X, y)

    stack = [(tree, 1)]

    while len(stack) > 0:
        node, depth = stack.pop()
        is_leaf = (node.__class__.__name__ == "QLeaf")

        output = ""
        output += "-" * depth
        output += " "

        if is_leaf:
            for leaf, X, y in leaf_samples_pairs:
                if leaf == node:
                    action = leaf.get_best_action()
                    matches = [(1 if action == y_i else 0) for y_i in y]
                    accuracy = np.mean(matches)
                    error = UCF(len(y), len(y) - np.sum(matches))
                    output += f"[bright_black]In-sample accuracy: {'{:.2f}'.format(100 - error * 100)}% ({len(y)} / {dataset_size}).[/bright_black]"
                
            # best_action_id = np.argmax(node.q_values)
            # output += (config['actions'][best_action_id]).upper()
        else:
            output += config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.3f}'.format(node.value)
            
            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))
        
        print(output)

    print("")
    avg, std = get_average_reward_with_std(
        config, tree,
        episodes=args['grading_episodes'],
        verbose=False)
    print(f"[yellow]--- Base reward for loaded model is {str_avg(avg, std)}.")
    
    for i, (leaf, X, y) in enumerate(leaf_samples_pairs):
        leaf.expert = expert

        print(f"[bright_black]- Leaf #{i} covers {len(X)} observations ({'{:.2f}'.format(len(X) / dataset_size * 100)}%). Replacing with expert...")
        avg, std = get_average_reward_with_std(
            config, tree,
            episodes=args['grading_episodes'],
            verbose=False)
        print(f"[yellow]--- Average reward for replacing leaf #{i} is {str_avg(avg, std)}.")

        leaf.expert = None

