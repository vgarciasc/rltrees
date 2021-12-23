class CTNode:
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
    
    def add_left_child(self, node):
        self.left = node
        
    def add_right_child(self, node):
        self.right = node

    def print_tree(self, level=1):
        print(" " * 2 * level, "if", f"x[{self.attribute}]", "<=", str(self.value) + ":")
        if self.left:
            self.left.print_tree(level + 1)
        print(" " * 2 * level, "else:")
        if self.right:
            self.right.print_tree(level + 1)

class CTLeaf:
    def __init__(self, action):
        self.action = action
    
    def print_tree(self, level=1):
        print(" " * 2 * level, str(self.action))
    
    def predict(self, state):
        return self.action

if __name__ == "__main__":
    tree = CTNode((0, 12), 
                CTNode((1, 1),
                    CTNode((0, 4), CTLeaf("left"), CTLeaf("right")),
                    CTLeaf("left")),
                CTNode((0, 4), 
                    CTNode((1, 2), CTLeaf("do nothing"), CTLeaf("left")),
                    CTNode((1, 1), CTLeaf("right"), CTLeaf("do nothing"))))
    tree.print_tree()
    print(tree.predict([2, 0]))
