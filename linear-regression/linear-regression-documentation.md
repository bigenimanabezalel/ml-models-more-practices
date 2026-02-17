---
Title: Linear Regression Concepts
Author: Jean Claude Bigenimana
---

# Regression

- The term regression is used when you try to find the relationship between variables.
- In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.

# Linear Regression

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