import pandas as pd
from os import getcwd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

"""
- In this code, we are performing linear regression on a dataset that contains information about the weight, volume, and CO2 emissions of certain objects. We first import the necessary libraries, including pandas for data manipulation, os for handling file paths, and sklearn for linear regression and data scaling.

- The scaling is the Z-score normalization given by (x - mean) / std where x is weight or volume, mean is the average of weight or volume and std is the standard deviation of weight or volume. This will give us a new matrix with the same number of rows and columns as X but with the values scaled to have a mean of 0 and a standard deviation of 1.

- The purpose of scaling the features is to ensure that they are on the same scale, which can improve the performance of the linear regression model. After scaling the features, we fit the linear regression model to the scaled data and print out the R^2 score, coefficients, and intercept of the model.

"""


scale = StandardScaler()
model = LinearRegression()

csv_path = getcwd() + "\\data\\data.csv"

df = pd.read_csv(csv_path)

X = df[["Weight", "Volume"]]
y = df["CO2"]

X_scaled = scale.fit_transform(X)

regr = model.fit(X_scaled, y)
predictions = regr.predict(X_scaled)

scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regr.predict([scaled[0]])

print(predictions)
print(predictedCO2)

print(f"The coefficients of the model are: {model.coef_}")
print(f"The value of the interecept is: {model.intercept_}")
print(f'The R^2 score is: {regr.score(X_scaled, y)}')
