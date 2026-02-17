


import matplotlib.pyplot as plt
from scipy import stats

# The example below is a scatter plot of the age and speed of 13 cars as they were passing a tollbooth.The x-axis represents age, and the y-axis represents speed. 

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# plt.scatter(x, y)
# plt.show()

# Add line of best fit to the scatter plot

'''linregress() method returns the slope, intercept, r-value, p-value and standard error of the estimated gradient. The line of best fit given by the defined fuunction is calculated using the slope and intercept. A list of predicted y values is created using the line of best fit and the x values. The line of best fit is then plotted on top of the scatter plot.'''

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))  # Same as mymodel = myfumc(i) for i in x. The results  is the predicted y values for each x value in the list x.

plt.scatter(x,y) 

plt.plot(x, mymodel) 

plt.savefig("line_scatter.png", dpi=150, bbox_inches="tight") 

plt.show()

# calculating the r-value , p-value and standard error of the estimated gradient and future values
print("R-value:", r_value)
print("P-value:", p_value)
print("Standard Error:", std_err)