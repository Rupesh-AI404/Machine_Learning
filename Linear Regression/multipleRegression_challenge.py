
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [4, 1],
    [5, 3],
    [6, 2]
], dtype=float)

y = np.array([12, 13, 19, 20, 28, 30], dtype=float)

# 1. Create the model
model = LinearRegression()

# 2. Train it using X and y
model.fit(X, y)


# 3. Print the intercept and coefficients
print(model.coef_)
print(model.intercept_)


# 4. Predict for a new observation: x1 = 7, x2 = 2
prediction = model.predict([[7, 2]])
print(prediction)


# 5. Calculate the training MSE
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(mse)

print(X.shape)


