# EDA and Data Cleaning

import pandas as pd

df = pd.read_csv('co2.csv')
df.head(10)
df.describe()

df.drop(['Make','Model','Vehicle Class','Transmission','Fuel Type'], axis = 1, inplace = True)

# Identifying Linear Relations with Visualization

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (20,10)

sb.pairplot(df)
plt.savefig('pairplor.png')

plt.scatter(x = 'Engine Size(L)', y = 'CO2 Emissions(g/km)', data = df, s = 100, alpha = 0.3, edgecolor = 'white')
plt.title('Engine size vs C02 Emissions', fontsize = 16)
plt.ylabel('CO2 Emissions', fontsize = 12)
plt.xlabel('Engine Size', fontsize = 12)
plt.savefig('enginesize_co2.png')

plt.scatter(x = 'Fuel Consumption Comb (L/100 km)', y = 'CO2 Emissions(g/km)', data = df, s = 100, alpha = 0.3, edgecolor = 'white')
plt.title('Fuel Consumption Comb (L/100 km) vs C02 Emissions', fontsize = 16)
plt.ylabel('CO2 Emissions', fontsize = 12)
plt.xlabel('Fuel Consumption Comb (L/100 km)', fontsize = 12)
plt.savefig('fcc_co2')

plt.scatter(x = 'Fuel Consumption Hwy (L/100 km)', y = 'CO2 Emissions(g/km)', data = df, s = 100, alpha = 0.3, edgecolor = 'white')
plt.title('Fuel Consumption Hwy (L/100 km) vs C02 Emissions', fontsize = 16)
plt.ylabel('CO2 Emissions', fontsize = 12)
plt.xlabel('Fuel Consumption City (L/100 km)', fontsize = 12)
plt.savefig('fch_co2.png')

plt.scatter(x = 'Fuel Consumption City (L/100 km)', y = 'CO2 Emissions(g/km)', data = df, s = 100, alpha = 0.3, edgecolor = 'white')
plt.title('Fuel Consumption City (L/100 km) vs C02 Emissions', fontsize = 16)
plt.ylabel('CO2 Emissions', fontsize = 12)
plt.xlabel('Fuel Consumption City (L/100 km)', fontsize = 12)
plt.savefig('fccity_co2.png')

# Simple Linear Regression model

X_var = df[['Engine Size(L)']] # independent variable
y_var = df['CO2 Emissions(g/km)'] # dependent variable

# 1. Statsmodel

import statsmodels.api as sm
from termcolor import colored as cl

X_var = sm.add_constant(X_var)

slr_model = sm.OLS(y_var, X_var) # Ordinary Least Squares 
slr_reg = slr_model.fit()

slr_reg.summary()

# 2. Scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)

lr = LinearRegression()
lr.fit(X_train, y_train)

yhat = lr.predict(X_test)

slr_slope = lr.coef_
slr_intercept = lr.intercept_

print(cl('R-Squared :', attrs = ['bold']), lr.score(X_test, y_test))

# Fitting slope and intercept 

sb.scatterplot(x = 'Engine Size(L)', y = 'CO2 Emissions(g/km)', data = df, s = 150, alpha = 0.3, edgecolor = 'white')
plt.plot(df['Engine Size(L)'], slr_slope*df['Engine Size(L)'] + slr_intercept, color = 'r', linewidth = 3)
plt.title('Engine size vs C02 Emissions', fontsize = 16)
plt.ylabel('CO2 Emissions', fontsize = 12)
plt.xlabel('Engine Size', fontsize = 12)
plt.savefig('enginesize_co2_fit.png')

# Mutiple Linear Regression 

X1_var = df[['Engine Size(L)','Fuel Consumption Comb (L/100 km)','Fuel Consumption Hwy (L/100 km)','Fuel Consumption City (L/100 km)']]

# 1. Statsmodel 

sm_X1_var = sm.add_constant(X1_var)

mlr_model = sm.OLS(y_var, sm_X1_var)
mlr_reg = mlr_model.fit()

mlr_reg.summary()

# 2. Scikit-learn

X_train, X_test, y_train, y_test = train_test_split(X1_var, y_var, test_size = 0.3, random_state = 0)

lr = LinearRegression()
lr.fit(X_train, y_train)

yhat = lr.predict(X_test)

print(cl('R-Squared :', attrs = ['bold']), lr.score(X_test, y_test))

# Visualizing Prediction Accuracy

sb.distplot(yhat, hist = False, color = 'r', label = 'Predicted Values')
sb.distplot(y_test, hist = False, color = 'b', label = 'Actual Values')
plt.title('Actual vs Predicted Values', fontsize = 16)
plt.xlabel('Values', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.legend(loc = 'upper left', fontsize = 13)
plt.savefig('ap.png')