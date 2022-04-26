import pdb
import pickle
from threading import activeCount
import matplotlib.pyplot as plt
import numpy as np
from TnTDecisionGraph.TreeInTree.TnT import TnT

class TnTWrapper:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y, pruning=0):
        clf = TnT(N1=2, N2=3)
        clf.fit(X, y)
        self.model = clf

    def act(self, state):
        state = state.reshape(1, -1)
        action = self.model.predict(state)
        action = action[0]
        return action

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved TnT to '{filename}'.")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        
    def get_size(self):
        return self.model.check_complexity

    # def get_as_qtree(self):
    #     children_left = self.model.tree_.children_left
    #     children_right = self.model.tree_.children_right
    #     feature = self.model.tree_.feature
    #     threshold = self.model.tree_.threshold
    #     values = self.model.tree_.value

    #     stack = [(0, 0, None)]
    #     output = []

    #     while len(stack) > 0:
    #         node_id, parent_id, is_left = stack.pop()

    #         is_leaf = children_left[node_id] == children_right[node_id]

    #         if is_leaf:
    #             q_values = values[node_id][0]
    #             output.append((node_id, "leaf", parent_id, is_left, q_values))
    #         else:
    #             split = (feature[node_id], threshold[node_id])
    #             output.append((node_id, "node", parent_id, is_left, split))

    #             stack.append((children_left[node_id], node_id, True))
    #             stack.append((children_right[node_id], node_id, False))

    #     output.sort(key = lambda x : x[0])
    #     return get_tree_from_print(output, self.config['actions'])

    # def get_as_viztree(self, show_prob=False):
    #     children_left = self.model.tree_.children_left
    #     children_right = self.model.tree_.children_right
    #     feature = self.model.tree_.feature
    #     threshold = self.model.tree_.threshold
    #     values = self.model.tree_.value

    #     stack = [(0, 1)]
    #     output = ""

    #     while len(stack) > 0:
    #         node_id, depth = stack.pop()
    #         is_leaf = children_left[node_id] == children_right[node_id]

    #         if is_leaf:
    #             content = self.config['actions'][np.argmax(values[node_id][0])].upper()
    #             if show_prob:
    #                 prob = np.max(values[node_id][0]) / sum(values[node_id][0])
    #                 content += f" ({'{:.2f}'.format(prob)})"
    #         else:
    #             content = self.config['attributes'][feature[node_id]][0] + " <= " + '{:.3f}'.format(threshold[node_id])

    #             stack.append((children_right[node_id], depth + 1))
    #             stack.append((children_left[node_id], depth + 1))

    #         output += f"\n{'-' * depth} {content}"

    #     return output