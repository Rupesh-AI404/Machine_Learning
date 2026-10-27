
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [4, 1],
    [5, 3],
    [6, 2]
], dtype=float)

y = np.array([12, 13, 19, 20, 28, 30], dtype=float)

# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 2. Train only on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Predict on the unseen test set
y_test_pred = model.predict(X_test)

# 4. Evaluate the test predictions
print("Actual:", y_test)
print("Predicted:", y_test_pred)
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
