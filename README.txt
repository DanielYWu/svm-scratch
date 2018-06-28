Author: Daniel Wu (5214001)
Email: wuxx1495@umn.edu

CSCI 5525 HW2

There are 3 executable python scripts:
    1. myDualSVM.py: Implementation of SVM in the non-separable case, using CVXOPT's quadratic optimization solver for the dual problem
    2. myPegasos.py: Implementation of the Pegasos algorithm for SVMs using stochastic gradient descent of the subgradient.
    3. mySoftPlus.py: Implementation of the Softplus estimation for cost function for SVMs and solve the optimization problem using stochastic gradient descent.

Required Libraries + versions for running python scripts:
    python 3.4
    numpy
    matplotlib
    sys (for file reading)
    sklearn (used cross_validation library for train-test split data, did not use model_selection since CSE linux desktop machines only have 1.7.x)

Requirements for the dataset:
    1. Must be in .csv format
    3. Row for cases, column for features
    4. The first column should be the target values, the rest of the columns are features of each data point.


Files can be run as directed in instructions from HW in terminal using python3
    python3 ./myDualSVM.py /path/to/dataset.csv C
        Example: python3 ./myDualSVM.py ./MNIST-13.csv 1e-3
    python3 ./myPegasos.py /path/to/dataset.csv k numruns
        Example: python3 ./myPegasos.py ./MNIST-13.csv 1000 5
    python3 ./mySoftPlus.py /path/to/MNIST-13.csv k numruns
        Example: python3 ./mySoftPlus.py ./MNIST-13.csv 1000 5


Other Notes:
    5. Features that had an STD of 0 in the dataset were omitted for computing SVMs.
    6. SVM implementation in these functions only support binary classification.