# Multiple Linear Regressions - data of 50 startups - to determine which company is the most
# interesting to invest and if there's correlation between profit, r&d, adminis, marketing and state
# MLR with backward elimination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_X = LabelEncoder()
X[:,3] = encoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Drop first aka avoiding dummy variable trap
X = X[:, 1:]

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MLR.fit into training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting results
y_predict = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
print(mean_absolute_error(y_test,y_predict))
print(mean_squared_error(y_test,y_predict))
print(np.sqrt(mean_squared_error(y_test,y_predict)))

# Building model using Backward Elimination, need an arr of 1, to imitate b0 in equation
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# ordinary least sqares Sl = 5% == 0,05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# x2 to remove

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
