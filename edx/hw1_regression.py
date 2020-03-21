import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter=",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter=",")

# transforming X

X_train, X_test = np.array(X_train), np.array(X_test)


## Solution for Part 1
def part1(y, X, lamb):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    wRR = np.dot(np.linalg.inv(np.dot(X.T, X) + (lamb * np.identity(X.shape[1]))),
                 np.dot(X.T, y))
    return wRR


wRR = part1(y=y_train, X=X_train, lamb=lambda_input)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")  # write output to file


## Solution for Part 2
def part2(lamb, sigma2, X_train, X_test, elem=10):
    # initializing the list for keeping indexes
    # creating a copy of initial test set to get initial indexes

    indexes = []
    init_indexes = X_test.copy()

    while len(indexes) <= elem:
        sigma2_0 = []

        # calculating posterior covariance
        cov = np.linalg.inv(np.dot(X_train.T, X_train) / sigma2 + (lamb * np.identity(X_train.shape[1])))
        for x in X_test:
            # calculating variance
            sigma2_0.append(sigma2 + np.dot(np.dot(x.T, cov), x))

        row_in_array = X_test[np.argmax(sigma2_0)]
        ind_init = np.where((init_indexes == row_in_array).all(axis=1))[0][0]
        indexes.append(ind_init + 1)
        X_train = np.append(X_train, X_test[np.argmax(sigma2_0)].reshape(-1, X_train.shape[1]), axis=0)
        X_test = np.delete(X_test, np.argmax(sigma2_0), axis=0)

    return np.array(indexes).astype(int)


active = part2(lamb=lambda_input, sigma2=sigma2_input, X_train=X_train,
               X_test=X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active,
           delimiter=",")  # write output to file