from read_file import *
import matplotlib.pyplot as plt
from myPegasos import myPegasos
from mySoftplus import mySoftplus
from myPegasos import featureNormalize


def draw_plot(axis, k):
    for i in range(5):
        mP = mySoftplus(X, y, 1e-4, k)
        
        axis.plot(mP.lossf)


X, y = read_file('MNIST-13.csv')
X = featureNormalize(X)


f, axarr = plt.subplots(3, 2, figsize=(8, 8))
axarr[0, 0].set_title('k = 1')
draw_plot(axarr[0, 0], 1)
axarr[0, 1].set_title('k = 20')
draw_plot(axarr[0, 1], 20)
axarr[1, 0].set_title('k = 200')
draw_plot(axarr[1, 0], 200)
axarr[1, 1].set_title('k = 1000')
draw_plot(axarr[1, 1], 1000)
axarr[2, 0].set_title('k = 2000')
draw_plot(axarr[2, 0], 2000)
plt.tight_layout()
# plt.savefig('../tex/figure/sgd.pdf', transparent=True, dpi=600)
plt.show()
