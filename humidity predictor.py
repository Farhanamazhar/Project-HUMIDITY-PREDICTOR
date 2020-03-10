# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Data.csv')
r=data.iloc[:,-1].values
y=r.reshape(-1,1)

m=data.iloc[:,-4].values
print(m)
b=[]
for i in m:
   f=i.replace('/','').replace(':','').replace(' ','')
   print(f)
   b.append(f)
   print(b)
x=np.array(b)   
x=x.reshape(-1,1)
x=list(map(float,x))
x=np.array(x)
x=x.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 12)
X_poly = poly_reg.fit_transform(x)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('DATE')
plt.ylabel('HUMIDITY')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('DATE')
plt.ylabel('HUMIDITY')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 100)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('DATE')
plt.ylabel('HUMIDITY')
plt.show()

# Predicting a new result with Polynomial Regression
a=input('Enter date as mddhhmm : ')
print("HUMIDITY at ",a,"=",lin_reg_2.predict(poly_reg.fit_transform([[a]])))
