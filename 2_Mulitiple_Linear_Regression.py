# Multiple Linear Regressions - data of 50 startups - to determine which company is the most
# interesting to invest and if there's correlation between profit, r&d, adminis, marketing and state

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:, 4].values

#pd.get_dummies vs OneHotEncoder
z = pd.get_dummies(X[:,3],drop_first=True)
z2 = z.values

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
