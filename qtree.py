import numpy as np

class QNode():
	def __init__(self, split, left = None, right = None):
		attribute, value = split
		self.attribute = attribute
		self.value = value

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
	
	def reset_qvalues(self):
		self.left.reset_qvalues() if self.left is not None else None
		self.right.reset_qvalues() if self.right is not None else None

	def reset_all(self):
		self.left.reset_all() if self.left is not None else None
		self.right.reset_all() if self.right is not None else None

class QLeaf():
	def __init__(self,  parent=None, is_left=False, actions=[]):
		self.actions = actions
		self.n_actions = len(actions)
		self.parent = parent
		self.is_left = is_left

		self.history = [[] for _ in range(self.n_actions)]
		self.q_history = [[] for _ in range(self.n_actions)]
		self.q_values = np.zeros(self.n_actions)

	def print_tree(self, level=1):
		# print(" " * 2 * level, f"Q(s, left)  = {self.q_values[0]}")
		# print(" " * 2 * level, f"Q(s, right) = {self.q_values[1]}")
		best_action_id = np.argmax(self.q_values)
		for action_id in range(self.n_actions):
			print(" " * 2 * level, f"Q(s, {self.actions[action_id]})  = {'{:.4f}'.format(self.q_values[action_id])}, mean ΔQ = {'---' if len(self.q_history[action_id]) == 0 else '{:.4f}'.format(np.mean([q for q in self.q_history[action_id]]))}, var ΔQ = {'---' if len(self.q_history[action_id]) == 0 else '{:.4f}'.format(np.var([q for q in self.q_history[action_id]]))} {' [*]' if best_action_id == action_id else ''}")
	
	def predict(self, state):
		return self, np.argmax(self.q_values)
	
	def reset_history(self):
		self.history = [[] for _ in range(self.n_actions)]
		self.q_history = [[] for _ in range(self.n_actions)]
	
	def reset_qvalues(self):
		self.q_values = np.zeros(self.n_actions)

	def reset_all(self):
		self.reset_history()
		self.reset_qvalues()
	
	def record(self, state, action, delta_q):
		self.q_values[action] += delta_q
		self.history[action].append((state, delta_q))
		self.q_history[action].append(delta_q)

def grow_tree(tree, leaf, splitting_criterion):
	split = splitting_criterion(leaf)

	new_node = QNode(split, None, None)
	new_node.left = QLeaf(parent=new_node, is_left=True, actions=leaf.actions)
	new_node.right = QLeaf(parent=new_node, is_left=False, actions=leaf.actions)

	if leaf.parent is None:
		return new_node

	if leaf.is_left:
		leaf.parent.left = new_node
	else:
		leaf.parent.right = new_node
	
	return tree