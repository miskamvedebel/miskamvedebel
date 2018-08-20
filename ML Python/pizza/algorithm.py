# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('pizza_dataset.csv', sep=';')
data_transformed = pd.DataFrame([i.split(',') for i in data['orders']], columns=['pizza1', 'pizza2', 'pizza3'])
data_transformed['amount'] = data['total_amount']

orders_sorted = orders.copy()
l = orders_sorted.values.tolist()
for a in l:
    a = a.sort()
X = pd.DataFrame(l)
X['synt'] = [str(X.loc[i, 0]) + str(X.loc[i, 1]) + str(X.loc[i, 2]) for i in range(len(X))]

y = data_transformed.loc[:, 'amount']

regressor = DecisionTreeRegressor(random_state=0)
X_train = pd.get_dummies(X['synt'])

regressor.fit(X_train, y)

d = np.array([[0, 0, 1], [2, 0, 0], [1, 0, 0]])

y_pred = regressor.predict(X=d)

linear = LinearRegression()
linear.fit(X, y)
y_pred = linear.predict(d)