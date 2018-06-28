import os.path
import numpy as np


def read_file(filename):
    assert os.path.isfile(filename)
    print("")
    print('Loading', filename)
    data = np.genfromtxt(filename, delimiter=",", skip_header=0)
    print("Imported successfully")
    print("")
    X = data[:, 1:]
    target = data[:, 0]
    y = np.zeros_like(target)
    for i in range(target.shape[0]):
        if target[i] == 3:
            y[i] = -1
        else:
            y[i] = target[i]
    return X, y
