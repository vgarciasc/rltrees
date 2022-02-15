import argparse
import pickle
import imitation_learning.env_configs
import numpy as np

from distilled_tree import DistilledTree
from qtree import QNode, QLeaf, save_tree_from_print

def get_qtree_structure_viz(config, qtree):
    stack = [(qtree, 1)]
    output = ""

    while len(stack) > 0:
        node, depth = stack.pop()
        is_leaf = (node.__class__.__name__ == "QLeaf")

        output += "\n"
        output += "-" * depth
        output += " "

        if is_leaf:
            best_action_id = np.argmax(node.q_values)
            output += (config['actions'][best_action_id]).upper()
        else:
            output += config['attributes'][node.attribute][0]
            output += " <= "
            output += '{:.3f}'.format(node.value)
            
            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decision Tree Visualization')
    parser.add_argument('-t','--task',help="Which task?", required=True)
    parser.add_argument('-f','--filepath',help="Which tree to view?", required=True)
    args = vars(parser.parse_args())

    config = imitation_learning.env_configs.get_config(args['task'])

    # dt = DistilledTree(config)
    # dt.load_model(args['filepath'])

    with open(args['filepath'], "rb") as f:
        qtree = pickle.load(f)
    
    print("===> Usual visualization:")
    qtree.print_tree()
    
    print("\n===> Web Tool Visualization:")
    print(get_qtree_structure_viz(config, qtree))