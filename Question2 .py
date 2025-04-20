"""Created on Sun Mar 12 22:41:04 2023

@author: SMAR
"""

## Import Libraries
import numpy as np
import pandas as pd

## Importing the dataset
dataset=pd.read_csv("50_Startups.csv")


## Remove the null value. “Replace it with most repeated value”.
#print (dataset)
dataset = dataset.fillna(dataset.mean())
print (dataset)


# Divide data to input and output
X = dataset.iloc[:, :-2].values
#print (X)
y = dataset.iloc[:, -1].values
#print (y)


##Encoding the string value
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
coltrans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(coltrans.fit_transform(X))
print(X)

## Split value of dataset (80% for training and 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


## Make a regression model to predict the test value.

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
y_pred2=np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
#print(y_pred2)


## Create a mean square error “MSE” function to calculate the error between true and predict values.
dataset=dataset.dropna()
X = dataset.iloc[:, :-2].values
#print (X)
y = dataset.iloc[:, -1].values
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred)
print (MSE)