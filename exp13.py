import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
import TnTDecisionGraph.TreeInTree as tnt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def get_tnt_viztree(model, feature_names, label_names):
    """
        similar to "graph_traverse", but with no data input
    """
    node_printed = []
    node_list = [(model.graph, 1)]
    output = []

    while len(node_list) > 0:
        n, depth = node_list.pop()
        
        if n.label is None:
            if n in node_printed:
                output.append(("-" * depth) + f" GOTO #{node_printed.index(n)}")
                # print(output[-1])
                continue
            else:
                output.append(("-" * depth) + f" #{len(node_printed)} {feature_names[n.feature_index]} <= {'{:.3f}'.format(n.threshold)}")
                node_printed.append(n)
       
            if n.left in node_list:
                node_list.remove(n.left)
            node_list.append((n.left, depth + 1))
            
            if n.right in node_list:
                node_list.remove(n.right)
            node_list.append((n.right, depth + 1))
        else:
            output.append(("-" * depth) + f" {label_names[n.label]}")
        
        # print(output[-1])
    
    return "\n".join(output)

if __name__ == "__main__":
    # df = datasets.load_iris()
    # X, y = df.data[:, :], df.target
    # feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    # label_names = ["Setosa", "Versicolor", "Virginica"]

    df = datasets.load_breast_cancer()
    X, y = df.data[:, :], df.target
    for _ in range(8):
        X = np.concatenate((X, X))
        y = np.concatenate((y, y))
    print(f"Dataset size: {len(X)}")
    feature_names = [f"{prefix} {feat}" for prefix in ["Mean", "SE", "Worst"] for feat in ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dimension"]]
    label_names = ["Malignant", "Benign"]

    # df = datasets.load_wine()
    # X, y = df.data[:, :], df.target
    # feature_names = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    # label_names = ["Class 0", "Class 1", "Class 2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train CART
    print("===> CART:")
    cart_model = DecisionTreeClassifier(ccp_alpha=0.005).fit(X_train, y_train)
    E_in = np.mean([(1 if y_i != y_train[i] else 0) for i, y_i in enumerate(cart_model.predict(X_train))])
    E_out = np.mean([(1 if y_i != y_test[i] else 0) for i, y_i in enumerate(cart_model.predict(X_test))])
    print(f"Generated CART has number of nodes: {cart_model.get_n_leaves() * 2 - 1}")
    print(f"Accuracy in-sample: {1-E_in}")
    print(f"Accuracy out-of-sample: {1-E_out}")
    print("")

    # Train TnT
    print("===> Tree-in-Tree:")
    tnt_model = tnt.TnT()

    start_time = time.time()
    tnt_model.fit(X_train, y_train)
    end_time = time.time()
        
    for _ in range(5):
        tnt_model.fix()
    
    viztree = get_tnt_viztree(tnt_model, feature_names[:], label_names)
    with open("data/tnt_viztree.txt", "w") as f:
        f.write(viztree)

    # pdb.set_trace()

    E_in = np.mean([(1 if y_i != y_train[i] else 0) for i, y_i in enumerate(tnt_model.predict(X_train))])
    E_out = np.mean([(1 if y_i != y_test[i] else 0) for i, y_i in enumerate(tnt_model.predict(X_test))])
    
    print(f"Generated TnT has complexity: {tnt_model.check_complexity()}")
    print(f"Elapsed time: {end_time - start_time} seconds")
    print(f"Accuracy in-sample: {1-E_in}")
    print(f"Accuracy out-of-sample: {1-E_out}")

    svg = tnt.utils.visTnT(tnt_model, X_train, y_train)
    tnt.svg.displaySVG(svg)