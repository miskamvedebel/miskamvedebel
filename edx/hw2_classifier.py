from __future__ import division
import numpy as np
import sys
from scipy.stats import multivariate_normal

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")


## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):
    '''return posterior probabilities'''

    N = len(y_train)
    classes, prior_probs = np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1] / N
    mean = [np.mean(X_train[np.where(y_train == c)], axis=0) for c in classes]
    cov = [np.cov(X_train[np.where(y_train == c)], rowvar=False) for c in classes]

    RV = [multivariate_normal(mean=m, cov=cov) for m, cov in zip(mean, cov)]

    post_prob = []

    for x_test in X_test:
        frac = prior_probs * np.array([rv.pdf(x_test) for rv in RV])
        probs = frac / np.sum(frac)
        post_prob.append(probs)

    return post_prob


final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file