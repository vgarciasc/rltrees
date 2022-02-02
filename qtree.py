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

def save_tree_from_print(printed_structure, actions, suffix):
	qtree = QNode(printed_structure[0][4], None, None, None)
	
	tree_list = [qtree]
	for id, kind, parent_id, is_left, essence in printed_structure[1:]:
		parent = tree_list[parent_id]

		if kind == "node":
			new_node = QNode(essence, parent, None, None)
		elif kind == "leaf":
			new_node = QLeaf(parent, is_left, actions, essence)

		if is_left:
			parent.left = new_node
		else:
			parent.right = new_node
		
		tree_list.append(new_node)

	save_tree(qtree, suffix)

if __name__ == "__main__":
	save_tree_from_print([
			(0, "node", None,  None,  (3, -0.696424812078476)),
			(1, "node", 0,     True,  (3, -0.700973224639892)),
			(2, "leaf", 1,     True,  [-23.0902, -25.0717, -17.6198, -22.7378]),
			(3, "leaf", 1,    False,  [-11.6993, -11.6450, -11.9116, -15.8909]),
			(4, "node", 0,    False,  (5, -0.003746651695109)),
			(5, "leaf", 4,     True,  [-23.4585, -21.2319, -21.7727, -24.2635]),
			(6, "node", 4,    False,  (5,  0.014616578817367)),
			(7, "leaf", 6,     True,  [-14.6318, -14.5968,  -9.3211, -15.0491]),
			(8, "leaf", 6,    False,  [-15.3591, -15.6862, -15.5153, -14.8529])],
		["nop", "left engine", "main engine", "right engine"],
		"_lunar_lander_v2_print")

	save_tree_from_print([
			(0, "node", None,  None,  (2, 0)),
			(1, "node", 0,    False,  (1, 8)),
			(2, "node", 1,     True,  (0, 17)),
			(3, "node", 1,    False,  (0, 18)),
			(4, "node", 0,     True,  (0, 16)),
			(5, "node", 4,     True,  (1, 6)),
			(6, "node", 5,     True,  (1, 3)),
			(7, "node", 6,     True,  (0, 12)),
			(8, "node", 6,    False,  (0, 11)),
			(9, "leaf", 2,     True,  [0, 1]),
			(10, "leaf", 2,   False,  [1, 0]),
			(11, "leaf", 3,    True,  [0, 1]),
			(12, "leaf", 3,   False,  [1, 0]),
			(13, "leaf", 4,   False,  [1, 0]),
			(14, "leaf", 5,   False,  [0, 1]),
			(15, "leaf", 7,    True,  [0, 1]),
			(16, "leaf", 7,   False,  [1, 0]),
			(17, "leaf", 8,    True,  [0, 1]),
			(18, "leaf", 8,   False,  [1, 0])],
		["stick", "hit"],
		"_blackjack_optimal")
	
	save_tree_from_print([
			(0, "node", None,  None,  (3, 0.44)),
			(1, "node", 0,     True,  (2, 0.01)),
			(2, "node", 0,    False,  (2, -0.41)),
			(3, "leaf", 1,    True,  [1, 0]),
			(4, "leaf", 1,   False,  [0, 1]),
			(5, "leaf", 2,    True,  [1, 0]),
			(6, "leaf", 2,   False,  [0, 1]),
		],
		["left", "right"],
		"_cartpole_silva_2020")
	
	save_tree_from_print([
			(0,  "node", None, None,  (5, 0.04)),
			(1,  "node", 0,    True,  (3, -0.35)),
			(2,  "node", 1,    True,  (5, -0.22)),
			(3,  "node", 2,    True,  (4, -0.04)),
			(4,  "node", 3,    True,  (6, 0)),
			(5,  "node", 4,    True,  (2, 0.32)),
			(6,  "node", 5,    True,  (1, -0.11)),
			(7,  "node", 6,    True,  (4, 0.15)),
			(8,  "node", 7,    True,  (0, -0.34)),
			(9,  "leaf", 0,   False,  [0, 0, 1, 0]),
			(10, "leaf", 1,   False,  [1, 0, 0, 0]),
			(11, "leaf", 2,   False,  [1, 0, 0, 0]),
			(12, "leaf", 3,   False,  [0, 1, 0, 0]),
			(13, "leaf", 4,   False,  [0, 0, 0, 1]),
			(14, "leaf", 5,   False,  [0, 0, 0, 1]),
			(15, "leaf", 6,   False,  [0, 1, 0, 0]),
			(16, "leaf", 7,   False,  [0, 1, 0, 0]),
			(18, "leaf", 8,   False,  [0, 1, 0, 0]),
			(17, "leaf", 8,    True,  [0, 0, 1, 0]),
		],
		["nop", "left engine", "main engine", "right engine"],
		"_lunarlander_silva_2020")
	
	save_tree_from_print([
			(0, "node", None, None, (4, -0.07604862377047539)),
			(1, "node", 0,    True, (2, -0.013069174252450466)),
			(2, "node", 0, False, (2, -0.09865037351846695)),
			(3, "node", 2, True, (5, -0.00998950470238924)),
			(4, "node", 3, True, (3, -0.5160131752490997)),
			(5, "node", 2, False, (6, 0.5)),
			(6, "node", 5, True, (1, 1.0102267265319824)),
			(7, "node", 6, True, (3, -0.10799521207809448)),
			(8, "node", 6, False, (5, 0.052217885851860046)),
			(9, "leaf", 1, True, [0, 0, 1, 0]),
			(10, "leaf", 1, False, [0, 1, 0, 0]),
			(11, "leaf", 4, True, [0, 0, 1, 0]),
			(12, "leaf", 4, False, [0, 0, 0, 1]),
			(13, "leaf", 3, False, [0, 0, 0, 1]),
			(14, "leaf", 7, True, [0, 0, 1, 0]),
			(15, "leaf", 7, False, [0, 0, 0, 1]),
			(16, "leaf", 8, True, [0, 1, 0, 0]),
			(17, "leaf", 8, False, [0, 0, 0, 1]),
			(18, "leaf", 5, False, [1, 0, 0, 0]),
		],
		["nop", "left engine", "main engine", "right engine"],
		"_lunarlander_optimal")