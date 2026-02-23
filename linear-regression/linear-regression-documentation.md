---
Title: Linear Regression Concepts
Author: Jean Claude Bigenimana
---

# Regression

- The term regression is used when you try to find the relationship between variables.
- In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.

# Linear Regression [line]

- Linear regression uses the relationship between the data-points to draw a straight line through all them.
- This line can be used to predict future values.
- In Machine Learning, predicting the future is very important

```python
# The example below is a scatter plot of the age and speed of 13 cars as they were passing a tollbooth.The x-axis represents age, and the y-axis represents speed. 

import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r_value, p_value, std_err  = stats.linregress(x,y)

# function that uses the slope and intercept values to return a new y-value

def myline(x):
    return slope*x + intercept  

y_preds= list(map(myline,x))     # This produce a list of predicted y-values

plt.scatter(x, y)  
plt.plot(x,y_preds)     # line of linear regression

plt.show()
```

# R for Relationship

It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything.

- This relationship - **the coefficient of correlation** - is called r.

- The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.

-  python and the Scipy module can compute this value  

```python
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r_value)  ## The result -0.76 shows that there is a relationship, not perfect, but it indicates that we could use linear regression in future predictions
```

# Predict Future Values

- try to predict the speed of a 10 years old car

```python

speed = myline(10)

print(speed)  # predicted a speed is 85.6
```

# Polynomial Regression [curved_line]

- In case the data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression
- Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points
- In the example below, we have registered 18 cars as they were passing a certain tollbooth.
- The  car's speed, and the time of day (hour) are recorded,
- R^2 value show the relationship between x,y. The stronger it is , indicate that model can be used to predict future value 
- The x-axis represents the hours of the day and the y-axis represents the speed:

```python
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3)) # polyfit → finds coefficients; poly1d  → builds model function;  model(x) → gives 
# 3 is the degree of polynomial
myline = numpy.linspace(1, 22, 100) 
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# To find the relationship between the values of the x- and y-axis
print("R_qsuared is:", r2_score(y, mymodel(x))) 

# To Predict Future Values

speed = mymodel(17)
print("The future speed value ", speed)
```

![alt text](image.png) 

The lines in the image shows the optimal degree of a polynomial

## General knowledge 

- Problem with R²: It always increases when you add more predictors (or higher polynomial degree), even if those predictors are useless. 
- A very complex model may have R² = 1 but still be overfitting,
- Adjusted R² fixes this by penalizing unnecessary complexity

#### Intuition

- If adding a predictor improves the model significantly, Adjusted R² increases

- If adding a predictor doesn’t help, Adjusted R² decreases

- This helps you avoid overfitting.


# Multiple regression

- Multiple regression is like linear regression, but with more than one independent value,
- The prediction of a value of dependent value (y) is based on two or more indepent variables(X).
- It is common to name the list of independent values with a upper case X, and the list of dependent values with a lower case y
- Useful methods from the <sklearn> module include: 
        . LinearRegression() method to create a linear regression object
        . This method has fit() method that takes the independent and dependent values as parameters and fills the regression object with data that describes the relationship
        . Also, has predict() method that use new data to fitted model to predict future value 
- The coefficient of indpentent variable (use coef_ ) is a factor that describes the relationship with an unknown variable (i.e the increase of one unit on idepented variable how much will increase or decrease the depentent variable)
Eg  y= a.W+ b.V --> a, b are coefficient of the regression

 Example: Let predict car CO2 emmision based on car weight and  volume

 ```python
 import pandas as pd
 from sklearn import linear_model


df = pd.read_csv("data.csv")
X = df [['Weight','Volume']] ## 
y = df['CO2']
regression_model = linear_model.LinearRegression (X,y)
fitted_model = regression_model.fit()    

# Pridict future value y on fitted model with new data

predictedCO2 = regression_model.predict([[2300, 1300]])

print(predictedCO2)
print(regr.coef_)   # This will generate the coefficients of each variable in the list of the independent variable
```

- If result array represents the coefficient values of weight and volume.
        Weight: 0.00755095
        Volume: 0.00780526
    - These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
    - And if the engine size (Volume) increases by 1cm3, the CO2 emission increases by 0.00780526g.


# Evaluate the performance of the model using R-squared

```python
print("R_squared is:", round(model.score(X, y), 4))

'''R_squared is: 0.3766 means that the model explains about 37.7% of the variance in y based on the features in X. This indicates that there are other factors influencing CO2 emissions that are not captured by the model, and it may not be a very strong predictor of CO2 emissions.'''

