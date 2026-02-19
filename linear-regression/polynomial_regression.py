import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 

'''
Polynomial regression is a type of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial. It is used when the data points do not follow a linear relationship, but instead follow a curved pattern.

Polynomial regression can be used to model complex relationships between variables and can provide a better fit for the data compared to linear regression. 

The degree of the polynomial can be adjusted to find the best fit for the data, but it is important to avoid overfitting by using too high of a degree. 

polfit() function from the numpy library can be used to finds coefficients, and the r2_score() function from the sklearn library can be used to evaluate the performance of the model.

poly1d() function can be used to create a polynomial function from the coefficients obtained from polyfit().'''

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(x, y, 3)) 

myline = np.linspace(1, 22, 100) 

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()

x_plot = np.linspace(min(x), max(x), 200)

plt.scatter(x, y, color="black", label="Data")

for d in [1,2,3,5]:
    model = np.poly1d(np.polyfit(x, y, d))
    plt.plot(x_plot, model(x_plot), label=f"Degree {d}") # The figure shows that degree 3 is the best fit for the data

plt.legend()
plt.show()

# To find the relationship between the values of the x- and y-axis
# R_squared = 0.94 means that the model is a good fit for the data. The closer the value is to 1, the better the model fits thedata.
print("R_squared is:", r2_score(y, mymodel(x)))

# Let Predict Future Values
speed = mymodel(17)
print("The future speed value ", speed) # speed value  88.87