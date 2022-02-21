import argparse
import pickle
import imitation_learning.env_configs
import numpy as np

from distilled_tree import DistilledTree
from qtree import QNode, QLeaf, save_tree_from_print

def load_viztree(filename):
    with open(filename, "r") as f:
        return f.read()
    
def viztree2qtree(config, string):
    actions = [a.lower() for a in config['actions']]
    attributes = [name.lower() for name, _, _, _ in config['attributes']]

    lines = string.split("\n")

    parents = [None for _ in lines]
    child_count = [0 for _ in lines]

    for line in lines:
        depth = line.rindex("- ") + 1
        content = line[depth:].strip()

        parent = parents[depth - 1] if depth > 1 else None
        is_left = (child_count[depth - 1] == 0) if depth > 1 else None
        
        is_leaf = content.lower() in actions

        if not is_leaf:
            attribute, threshold = content.split(" <= ")
            
            attribute = attributes.index(attribute.lower())
            threshold = float(threshold)
            split = (attribute, threshold)

            node = QNode(split, parent)
        if is_leaf:
            action = actions.index(content.lower())

            q_values = np.zeros(len(actions))
            q_values[action] = 1

            node = QLeaf(parent, is_left, actions, q_values)
        
        if parent:
            if is_left:
                parent.left = node
            else:
                parent.right = node
        else:
            root = node

        parents[depth] = node
        child_count[depth] = 0
        child_count[depth - 1] += 1
    
    return root

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
    parser.add_argument('-o','--output',help="Where to output?", required=False, default="")
    args = vars(parser.parse_args())

    config = imitation_learning.env_configs.get_config(args['task'])

    # dt = DistilledTree(config)
    # dt.load_model(args['filepath'])

    with open(args['filepath'], "rb") as f:
        qtree = pickle.load(f)
    
    print("===> Usual visualization:")
    qtree.print_tree()
    
    print("\n===> Web Tool Visualization:")
    viztree = get_qtree_structure_viz(config, qtree)
    print(viztree)

    if args['output']:
        with open(args['output'], "w") as f:
            f.write(viztree)