from numpy import *
from read_file import read_file
from myDualSVM import *
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


def draw_plot(axis, C, prop):
    axis.semilogx(C, prop)


if __name__ == "__main__":
    X, y = read_file('MNIST-13.csv')
    X = featureNormalize(X)
    C = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    numsupportvectors = []
    geomargin = []
    errorRates = []

    for c in C:
        numsupportvectors_c = []
        geomargin_c = []
        errorRates_c = []
        # Run 80-20 train-test split n-times where n is the directed input
        for i in range(10):
            print('Round ', i + 1, ':')
            # perform the 80-20 split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2)
            alpha = optimize(X_train, y_train, c)
            w, b = classifier(alpha, X_train, y_train)
            # print('1/|w| = ', 1 / np.dot(w.T, w))
            geomargin_c.append(1 / np.dot(w.T, w))
            s_vect = (np.dot(X_train, w) + b) * y_train

            # print('support vectors: ', np.shape(
        # s_vect[np.round(s_vect, decimals=2) == 1.0]))
            numsupportvectors_c.append(
                s_vect[np.round(s_vect, decimals=2) == 1.0].shape[0])
            #numsupportvectors_c.append(alphas[alphas > 1e-6].shape[0])
            error = predict(w, b, X_test, y_test)
            # print('% test error: ', error)
            errorRates_c.append(error)
            print("\n")
        # Print combined error rates for
        # each train set percent averaged by the number of folds that ran
        print('Combined Error Rates for C=', c, ': ')
        print('% Mean Error Rate: ', np.mean(errorRates_c))
        print('% STD of Error Rate: ', np.std(errorRates_c), '\n')
        geomargin.append(np.mean(geomargin_c))
        errorRates.append(np.mean(errorRates_c))
        numsupportvectors.append(np.mean(numsupportvectors_c))
        print('# of support vectors: ', np.mean(numsupportvectors_c))
        print('1/|w| = ', np.mean(geomargin_c), '\n')

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 0].set_title('Geographical Margin vs. C')
    draw_plot(axarr[0, 0], C, geomargin)
    axarr[0, 1].set_title('# of Support Vectors vs. C')
    draw_plot(axarr[0, 1], C, numsupportvectors)
    axarr[1, 0].set_title('Test Error Rate vs. C')
    draw_plot(axarr[1, 0], C, errorRates)
    plt.tight_layout()
    # plt.savefig('../tex/figure/sgd.pdf', transparent=True, dpi=600)
    plt.show()
    print(geomargin)
    print(errorRates)
    print(numsupportvectors)
    
