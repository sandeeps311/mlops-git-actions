# Import modules and packages
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(6, 5))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    # plt.legend(shadow='True')
    # Set grids
    # plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    # Some text
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    # Show
    plt.savefig('model_results.png', dpi=120)



## Create features
X = np.arange(-100, 100, 4).reshape(-1, 1)

# Create labels
y = np.arange(-90, 110, 4)


# Split data into train and test sets
N = 25
X_train = X[:N] # first 40 examples (80% of data)
y_train = y[:N]

X_test = X[N:] # last 10 examples (20% of data)
y_test = y[N:]


# Create XGBoost model
model = XGBRegressor(random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_preds = model.predict(X_test)

# Plot predictions
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_preds)

# Calculate model metrics
mae_1 = mean_absolute_error(y_test, y_preds)
mse_1 = mean_squared_error(y_test, y_preds)
print(f'\nMean Absolute Error = {mae_1:.2f}, Mean Squared Error = {mse_1:.2f}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1:.2f}, Mean Squared Error = {mse_1:.2f}.')
