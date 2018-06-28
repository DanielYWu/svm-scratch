
import numpy as np
from cvxopt import matrix, solvers
from read_file import read_file
from sklearn.cross_validation import train_test_split
import sys

# X is the data matrix (each row is a data point)
# Y is desired output (1 or -1)


def optimize(X, y, Con):
    """estimate the alphas given X, y, Con
        Args:
            X (numpy.array): Data Matrix
            y (numpy.array): Response Vector
            Con: C constant on xi's
        Returns:
            numpy.array: the alphas that solve the dual problem
     """
    m = X.shape[0]  # number of datapoints
    n = X.shape[1]  # number of features
    M = y[:, None] * X
    P = matrix(np.dot(M, M.T))

    q = matrix(-np.ones((m, 1)))

    G1 = -1 * np.eye(m)
    G2 = 1 * np.eye(m)
    h1 = np.zeros(m)
    h2 = np.zeros(m)
    h2.fill(Con)

    newG = matrix(np.concatenate((G1, G2), axis=0))
    new_h = np.concatenate((h1[:, None], h2[:, None]), axis=0)
    newh = matrix(np.squeeze(new_h))

    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    # find the solution
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, newG, newh, A, b)
    xvals = np.array(solution['x'])

    return xvals


def classifier(alphas, X, y):
    """estimate the weight vector and intercept given alphas, X, y
        Args:
            alphas (numpy.array): Solution from the dual problem
            X (numpy.array): Data Matrix
            y (numpy.array): Response Vector
        Returns:
            numpy.array: trained weight vector w
            float: intercept b
    """

    w = np.sum(alphas * y[:, None] * X, axis=0)
    temp = np.dot(X, w)
    b1 = np.amax(temp[y == -1])

    b2 = np.amin(temp[y == 1])

    b = -1 / 2 * (b1 + b2)
    return w, b


def featureNormalize(X):
    """Preprocesses the data by subtracing the mean and dividing over std

        Args:
            X (numpy.array): Data Matrix (m x n)

        Returns:
            numpy.array: Processed Data Matrix
    """
    stds = X.std(axis=0)
    newX = np.delete(X, np.where(stds == 0), 1)

    stds = newX.std(axis=0)
    means = newX.mean(axis=0)
    return (newX - means) / stds


def predict(w, b, X, y):
    """predict the class label for each observation in the new Design Matrix X
    Args:
        w,b: Parameters of prediction
        X,y: Dataset
    Returns:
        numpy.array: vector of predicted class label
    """
    prediction = np.dot(w[:, None].T, X.T) + b
    prediction = np.squeeze(prediction)
    # print(prediction)
    predict = np.zeros_like(y)
    for i in range(prediction.shape[0]):
        if prediction[i] > 0:
            predict[i] = 1
        else:
            predict[i] = -1

    errors = np.array([predict[i] != y[i] for i in range(0, y.shape[0])])
    totalerror = np.sum(errors) / y.shape[0]
    return totalerror


if __name__ == "__main__":
    X, y = read_file(sys.argv[1])
    X = featureNormalize(X)
    C = float(sys.argv[2])

    errorRates = []

    totalerror = 0

    # Run 80-20 train-test split n-times where n is the directed input
    for i in range(10):
        print("Training model ", i + 1, ' out of 10')
        # perform the 80-20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2)
        alphas = optimize(X_train, y_train, C)
        w, b = classifier(alphas, X_train, y_train)

        s_vect = (np.dot(X_train, w) + b) * y_train

        error = predict(w, b, X_test, y_test)

        errorRates.append(error)

    # Print combined error rates for each train set
    # percent averaged by the number of folds that ran
    print("\n")
    print("-------FINAL RESULT -------")
    print('Combined Error Rates:')
    currentTrainset = []
    for i in range(10):
        currentTrainset.append(errorRates[i])
    print('% Mean Error Rate: ', np.mean(currentTrainset))
    print('% STD of Error Rate: ', np.std(currentTrainset))
