# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('for_algorithm.csv', sep=';')
X = data.iloc[:, :25].values
y = data.iloc[:, 26].values

d = np.identity(25)
# linear regression
linear = LinearRegression()
linear.fit(X, y)
y_pred = linear.predict(d)

# classification:

y_class = data.iloc[:, 25].values

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X, y_class)

ctry_ = np.random.randint(0, 26, size=(20, 3))
ctry_ = pd.DataFrame(ctry_)
for c in ctry_.columns:
    ctry_[c] = ctry_[c].map(mapping)
ctry = pd.DataFrame(data=np.zeros(shape=(len(ctry_), len(sparse_columns))), columns=sparse_columns)
for i in range(len(ctry_)):
    for pizza in list(ctry_.loc[i, :]):
        if pizza in sparse_columns:
            ctry.loc[i, pizza] += 1
y_test = ctry.iloc[:, :].values

y_pred_class = log_reg.predict(y_test)
y_pred_class_label = pd.Series(y_pred_class)
y_pred_class_label = y_pred_class_label.map(country_map)

proba = log_reg.predict_proba(y_test)

outcome = pd.DataFrame(ctry_)
outcome['predicted_country'] = y_pred_class_label