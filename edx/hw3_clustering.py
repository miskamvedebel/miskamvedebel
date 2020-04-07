import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
import matplotlib.pyplot as plt

def KMeans(data, K=5, iterations=10, replace=False):

    LOSS = []
    MUs = data[np.random.choice(len(data), size=K, replace=replace), :]

    # calculate distinces:
    for i in range(iterations):
        dd = [np.linalg.norm(data - MUs[k], axis=1) for k in range(K)]
        dist = np.stack(dd, axis=1)
        classes = np.argmin(dist, axis=1)
        LOSS.append(np.sum([dist[x, np.argmin(dist[x, :])] for x in range(dist.shape[0])]))
        mmu = [np.mean(data[classes == k], axis=0) for k in range(K)]
        MUs = np.stack(mmu, axis=0)
        # filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        # np.savetxt(filename, MUs, delimiter=",")
    return classes, MUs


def EMGMM(data, K=5, iterations=10):
    # init
    N = len(data)
    shape = data.shape[1]
    pi_k = np.ones(K) / K
    mu_ci = data[np.random.choice(len(data), size=K, replace=False), :]
    cov = [np.identity(shape) for k in range(K)]
    RVs = [multivariate_normal(mean=mu_ci[k, :], cov=cov[k]) for k in range(K)]
    for i in range(iterations):
        phi_i = []
        for xi in data:
            phi_i.append(np.nan_to_num(
                pi_k * np.array([rv.pdf(xi) for rv in RVs]) / np.sum(pi_k * np.array([rv.pdf(xi) for rv in RVs]))))
        phi_i = np.array(phi_i)
        n_k = np.sum(phi_i, axis=0)
        pi_k = n_k / N
        mu_k = (np.dot(phi_i.T, data).T / n_k).T
        cov_k = []
        for k in range(K):
            arr = np.zeros(shape=(8, 8))
            for ind, xi in enumerate(X):
                arr += phi_i[ind, k] * (np.outer(xi - mu_k[k], xi - mu_k[k]))
            cov_k.append(arr / n_k[k])
        RVs = [multivariate_normal(mean=mu_k[k, :], cov=cov_k[k]) for k in range(K)]

        filename = f"pi-{i + 1}.csv"
        np.savetxt(filename, pi_k, delimiter=",")
        filename = "mu-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu_k, delimiter=",")  # this must be done at every iteration

        for k in range(K):  # k is the number of clusters
            filename = "Sigma-" + str(k + 1) + "-" + str(
                i + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, cov_k[k], delimiter=",")

    return mu_k, phi_i