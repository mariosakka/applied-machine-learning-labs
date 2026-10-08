from sklearn.neural_network import MLPClassifier
import numpy as np

X = [[1., 0.], [2, 0.], [3, 0], [0, 1], [0,2], [0,0]]
T = [0, 0, 1, 0, 1, 1]

# net = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=([5,5]), random_state=1)

net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=([10,10]), random_state=1)
net.fit(X, T)

import matplotlib.pyplot as plt

p=[x[0] for x in X]
k=[x[1] for x in X]

plt.plot([-1, 4], [0, 0], 'k')
plt.plot([0, 0], [-1, 3], 'k')


for sample in range(len(T)):
    if T[sample]:
        plt.plot(p[sample], k[sample], 'ro',markersize=10)
    else:
        plt.plot(p[sample], k[sample], 'go',markersize=10)


for x in np.arange (-1,4,.1):
    for y in np.arange (-1,3,.1):
        if net.predict([[x,y]])[0]:
            plt.plot(x,y,'ro', markersize=5)
        else:
            plt.plot(x,y,'go', markersize=5)
            

plt.show()