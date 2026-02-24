
import pandas as pd
from sklearn.linear_model import LinearRegression


# Load the dataset
df = pd.read_csv(r'H:\Data Integrity Analyst\Data Science\JC\DATA ENGENEERING ACADEMY\traditional_models\data\data.csv')

X = df[['Weight', 'Volume']]
y = df['CO2']

# Create a linear regression model
model = LinearRegression() 
model.fit(X, y) 

# Get the coefficients
print("Coefficients:", round(model.coef_[0], 4), round(model.coef_[1], 4))  # Coefficients: [0.0076 0.0078]

# Get the intercept
print("Intercept:", round(model.intercept_, 4))    # Intercept: 79.6947

# Predict the CO2 emissions for a new car with a weight of 3000 and a volume of 1500
new_car = [[3000, 1500]]    
predicted_co2 = model.predict(new_car)

print("Predicted CO2 emissions for the new car:", predicted_co2[0])

# Evaluate the performance of the model using R-squared

print("R_squared is:", round(model.score(X, y), 4))

'''R_squared is: 0.3766 means that the model explains about 37.7% of the variance in y based on the features in X. This indicates that there are other factors influencing CO2 emissions that are not captured by the model, and it may not be a very strong predictor of CO2 emissions.'''


