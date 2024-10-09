import os
import pickle as pkl


def save(name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pkl.dump(name, f)


def load(path):
    with open(path, "rb") as f:
        return pkl.load(f)
