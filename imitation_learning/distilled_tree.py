import pdb
import pickle
import matplotlib.pyplot as plt
from sklearn import tree

class DistilledTree:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y, pruning=0):
        clf = tree.DecisionTreeClassifier(ccp_alpha=pruning)
        clf = clf.fit(X, y)
        self.model = clf

    def act(self, state):
        state = state.reshape(1, -1)
        action = self.model.predict(state)
        action = action[0]
        return action
    
    def save_fig(self):
        plt.figure(figsize=(25, 25))
        feature_names = [name for (name, _, _, _) in self.config["attributes"]]
        tree.plot_tree(self.model, feature_names=feature_names)
        plt.savefig('last_tree.png')

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved tree to '{filename}'.")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)

    def get_as_qtree(self):
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        values = self.model.tree_.value

        stack = [(0, 0, None)]
        output = []

        while len(stack) > 0:
            node_id, parent_id, is_left = stack.pop()

            is_leaf = children_left[node_id] == children_right[node_id]

            if is_leaf:
                q_values = values[node_id][0]
                output.append((node_id, "leaf", parent_id, is_left, q_values))
            else:
                split = (feature[node_id], threshold[node_id])
                output.append((node_id, "node", parent_id, is_left, split))

                stack.append((children_left[node_id], node_id, True))
                stack.append((children_right[node_id], node_id, False))

        output.sort(key = lambda x : x[0])
        return output

    def get_as_viztree(self):
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        values = self.model.tree_.value

        stack = [(0, 1)]
        output = []

        while len(stack) > 0:
            node_id, depth = stack.pop()

            is_leaf = children_left[node_id] == children_right[node_id]
            if is_leaf:
                pdb.set_trace()
                content = self.config['actions'][values[node_id][0]]
            else:
                content = self.config['attributes'][feature[node_id]][0] + " <= " + '{:.3f}'.format(threshold[node_id])

                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

            output.append(f"\n{'-' * depth} {content}")

        return output