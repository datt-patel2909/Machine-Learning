from sklearn.linear_model import LinearRegression
import numpy as np


X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])


predictions = model.predict(X)
print("Predictions:", predictions)


import matplotlib.pyplot as plt

plt.scatter(X, Y, color='blue')
plt.plot(X, predictions, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.show()
print("R² Score:", model.score(X, Y))

