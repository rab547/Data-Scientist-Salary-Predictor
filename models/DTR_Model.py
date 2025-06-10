import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataAnalytics')))
from dataAnalysis import *

# Importing and Cleaning the data
df = loadDataFrame("Data/ds_salaries.csv")
df = replaceNullsWithMedian(df)

X = df.drop(["salary_in_usd", "salary", "salary_currency"], axis=1)
y = df["salary_in_usd"]

ordinal_cols = ["company_size"]
ordinal_order = [["S", "M", "L"]]
X[ordinal_cols] = OrdinalEncoder(categories=ordinal_order).fit_transform(X[ordinal_cols])

X = pd.get_dummies(X, columns=["experience_level", "employment_type", "job_title", "employee_residence", "company_location"], drop_first=True)

# Splitting data to Train, Test, Validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Training Model
regressor_1 = DecisionTreeRegressor(max_depth=None, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
regressor_2 = DecisionTreeRegressor(max_depth=10, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
regressor_3 = DecisionTreeRegressor(max_depth=5, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
regressor_1.fit(X_train, y_train)
regressor_2.fit(X_train, y_train)
regressor_3.fit(X_train, y_train)
y_1 = regressor_1.predict(X_test)
y_2 = regressor_2.predict(X_test)
y_3 = regressor_3.predict(X_test)

# Finding the best model initially
# param_grid = {
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]
# }

# grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)


# best_model = grid_search.best_estimator_

# predictions = regressor.predict(X_test)

# Calculating performance
mse = mean_squared_error(y_test, y_1)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_1)
r2 = r2_score(y_test, y_1)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualizations
# Depths
depths = [3, 5, 10, None]
train_errors = []
val_errors = []

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    val_errors.append(mean_squared_error(y_val, val_pred))

plt.plot([str(d) for d in depths], train_errors, label="Train MSE", marker='o')
plt.plot([str(d) for d in depths], val_errors, label="Validation MSE", marker='o')
plt.xlabel("max_depth")
plt.ylabel("Mean Squared Error")
plt.title("Model Accuracy vs. Tree Depth")
plt.legend()
plt.show()

# Num of Features
num_features = [0.5, 0.75, 1.0]
train_errors = []
val_errors = []

for num in num_features:
    X = X.sample(frac=num, axis='columns')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = DecisionTreeRegressor(max_depth=10, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    val_errors.append(mean_squared_error(y_val, val_pred))

plt.plot([str(num) for num in num_features], train_errors, label="Train MSE", marker='o')
plt.plot([str(num) for num in num_features], val_errors, label="Validation MSE", marker='o')
plt.xlabel("percent of features used")
plt.ylabel("Mean Squared Error")
plt.title("Model Accuracy vs. Number of Features (fractions)")
plt.legend()
plt.show()



min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)
X_val_minmax = min_max_scaler.fit_transform(X_val)

train_errors = []
val_errors = []

model = DecisionTreeRegressor(max_depth=10, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)

model.fit(X_train, y_train)
normal_model = DecisionTreeRegressor(max_depth=10, max_features=None, min_samples_leaf=4, min_samples_split=10, random_state=42)
normal_model.fit(X_train_minmax, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
normal_train_pred = normal_model.predict(X_train_minmax)
normal_val_pred = normal_model.predict(X_val_minmax)

train_errors.append(mean_squared_error(y_train, train_pred))
train_errors.append(mean_squared_error(y_train, normal_train_pred))

val_errors.append(mean_squared_error(y_val, val_pred))
val_errors.append(mean_squared_error(y_val, normal_val_pred))

plt.plot(["not normalized", "normalized"], train_errors, label="Train MSE", marker='o')
plt.plot(["not normalized", "normalized"], val_errors, label="Validation MSE", marker='o')
plt.xlabel("normalized?")
plt.ylabel("Mean Squared Error")
plt.title("Model Accuracy vs. Normalization")
plt.legend()
plt.show()