
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Prepare the training data
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([15, 25, 35, 45, 55], dtype=float)

# 2. Create the model
model = LinearRegression()

# 3. Learn the best-fit line from the data
model.fit(X, y)

# 4. Inspect the learned parameters
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# 5. Predict for a new input
prediction = model.predict([[6]])
print("Prediction at X=6:", prediction)

# 6. Calculate training predictions
y_pred = model.predict(X)

# 7. Calculate Mean Squared Error
mse = np.mean((y - y_pred) ** 2)
print("Training MSE:", mse)
