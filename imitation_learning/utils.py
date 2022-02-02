from rich import print
import pickle

def printv(str, verbose=True):
    if verbose:
        print(str)

def save_dataset(filename, X, y):
    with open(filename, "wb") as f:
        pickle.dump((X, y), f)

def load_dataset(filename):
    with open(filename, "rb") as f:
        X, y = pickle.load(f)
    return X, y