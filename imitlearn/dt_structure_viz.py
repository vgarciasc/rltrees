import argparse
import pickle
import imitlearn.env_configs
import numpy as np

from imitlearn.distilled_tree import DistilledTree
from imitlearn.ova import CartOvaAgent
from qtree import QNode, QLeaf, load_tree, save_tree_from_print

def load_viztree(filename):
    with open(filename, "r") as f:
        return f.read()
    
def viztree2qtree(config, string, expert=None):
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
        
        is_leaf = "<=" not in content

        if not is_leaf:
            attribute, threshold = content.split(" <= ")
            
            attribute = attributes.index(attribute.lower())
            threshold = float(threshold)
            split = (attribute, threshold)

            node = QNode(split, parent)
            
        if is_leaf:
            q_values = np.zeros(len(actions))
            leaf_expert = None

            if content.lower() == "expert":
                leaf_expert = expert
            else:
                action = actions.index(content.lower())
                q_values[action] = 1

            node = QLeaf(parent, is_left, actions, 
                q_values=q_values, expert=leaf_expert)
        
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
    parser.add_argument('-c','--class',help="What is the tree's class?", required=True)
    parser.add_argument('-f','--filepath',help="Which tree to view?", required=True)
    parser.add_argument('-o','--output',help="Where to output?", required=False, default="")
    args = vars(parser.parse_args())

    config = imitlearn.env_configs.get_config(args['task'])

    if args['class'] == "DistilledTree":
        dt = DistilledTree(config)
        dt.load_model(args['filepath'])
        viztree = dt.get_as_viztree()
    elif args['class'] == "CartOva":
        dt = CartOvaAgent(config)
        dt.load_model(args['filepath'])
        dt.prune_redundant_leaves()
        viztree = dt.get_as_viztree()
    elif args['class'] == "QTree":
        dt = load_tree(args['filepath'])
        viztree = get_qtree_structure_viz(config, dt)
    
    print("\n===> Web Tool Visualization:")
    print(viztree)

    if args['output']:
        with open(args['output'], "w") as f:
            f.write(viztree)