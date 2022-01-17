import scipy.stats as stats
import numpy as np
import pdb
from scipy.stats.mstats_basic import kurtosis, skew
from datetime import datetime
import pickle

class QNode():
	def __init__(self, split, parent=None, left=None, right=None):
		attribute, value = split
		self.attribute = attribute
		self.value = value

		self.parent = parent
		self.left = left
		self.right = right
	
	def predict(self, state):
		if state[self.attribute] <= self.value:
			return self.left.predict(state)
		else:
			return self.right.predict(state)
	
	def set_left_child(self, node):
		self.left = node
		
	def set_right_child(self, node):
		self.right = node

	def print_tree(self, level=1):
		print(" " * 2 * level, "if", f"x[{self.attribute}]", "<=", str(self.value) + ":")
		if self.left:
			self.left.print_tree(level + 1)
		print(" " * 2 * level, "else:")
		if self.right:
			self.right.print_tree(level + 1)
	
	def reset_history(self):
		self.left.reset_history() if self.left is not None else None
		self.right.reset_history() if self.right is not None else None
	
	def reset_values(self):
		self.left.reset_values() if self.left is not None else None
		self.right.reset_values() if self.right is not None else None

	def reset_all(self):
		self.left.reset_all() if self.left is not None else None
		self.right.reset_all() if self.right is not None else None
	
	def __str__(self):
		return f"x[{self.attribute}] <= {str(self.value)}"
	
	def get_size(self):
		return 1 + self.left.get_size() + self.right.get_size()
	
	def get_leaves(self):
		return [] + self.left.get_leaves() + self.right.get_leaves()

class QLeaf():
	def __init__(self,  parent=None, is_left=False, actions=[], q_values=None, value=None):
		self.actions = actions
		self.n_actions = len(actions)
		self.parent = parent
		self.is_left = is_left

		self.reset_all()

		self.q_values = np.zeros(self.n_actions) if q_values is None else q_values
		self.value = 0 if value is None else value

	def print_tree(self, level=1):
		# print(" " * 2 * level, f"Q(s, left)  = {self.q_values[0]}")
		# print(" " * 2 * level, f"Q(s, right) = {self.q_values[1]}")
		best_action_id = np.argmax(self.q_values)
		for action_id in range(self.n_actions):
			test = "---"
			if len(self.dq_history[action_id]) > 100:
				data = self.dq_history[action_id][-100:]
				value = 2 * np.std(data, ddof=1)
				test = ("test: " + str(value)) 
			print(" " * 2 * level, f"Q(s, {self.actions[action_id]})  = {'{:.4f}'.format(self.q_values[action_id])}, mean ΔQ = {'---' if len(self.dq_history[action_id]) == 0 else '{:.4f}'.format(np.mean([q for q in self.dq_history[action_id]]))}, var ΔQ = {'---' if len(self.dq_history[action_id]) == 0 else '{:.4f}'.format(np.var([q for q in self.dq_history[action_id]]))}, {test}, {' [*]' if best_action_id == action_id else ''}")
	
	def predict(self, state):
		if np.sum(self.q_values) == 0:
			return self, np.random.randint(0, self.n_actions)
		return self, np.argmax(self.q_values)
	
	def reset_history(self):
		self.state_history = []
		self.dq_history = [[] for _ in range(self.n_actions)]
		self.full_dq_history = [[] for _ in range(self.n_actions)]
		self.q_history = [[] for _ in range(self.n_actions)]
		self.full_q_history = [[] for _ in range(self.n_actions)]
	
	def reset_values(self):
		self.value = 0
		self.q_values = np.zeros(self.n_actions)

	def reset_all(self):
		self.reset_history()
		self.reset_values()
	
	def record(self, state, action, delta_q):
		self.q_values[action] += delta_q

		self.state_history.append(state)
		self.dq_history[action].append(delta_q)
		self.full_dq_history[action].append((state, delta_q))
		self.q_history[action].append(self.q_values[action])
		self.full_q_history[action].append((state, self.q_values[action]))
	
	def get_size(self):
		return 1
	
	def get_leaves(self):
		return [self]

def grow_tree(tree, leaf, splitting_criterion, split=None):
	if split is None:
		split = splitting_criterion(leaf)

	new_node = QNode(split, leaf.parent, None, None)
	new_node.left = QLeaf(parent=new_node, is_left=True, actions=leaf.actions)
	new_node.right = QLeaf(parent=new_node, is_left=False, actions=leaf.actions)

	if leaf.parent is None:
		return new_node

	if leaf.is_left:
		leaf.parent.left = new_node
	else:
		leaf.parent.right = new_node
	
	return tree

def save_tree(tree, suffix="", reward=-1):
	filename = datetime.now().strftime("tree %Y-%m-%d %H-%M")
	with open('data/' + filename + suffix, 'wb') as file:
		pickle.dump(tree, file)
		print(f"> Saved tree of size {tree.get_size()} " + (f"and reward {reward} " if reward != -1 else "") + "to file 'data/" + filename + suffix + "'!")

