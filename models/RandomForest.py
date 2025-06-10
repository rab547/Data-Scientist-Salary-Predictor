import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.base import clone
import joblib


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

joblib.dump(X_train.columns.tolist(), "models/saved_models/feature_columns.pkl")

common_params = dict(
    max_depth=3,
    min_samples_leaf=4,
    min_samples_split=4,
)

param_grid = dict(
    n_estimators=[100, 150, 200, 250, 300],
    max_depth=[2, 5, 10, 15, 20],
    min_samples_leaf=[1, 5, 10, 20, 30, 50],
    min_samples_split=[2, 5, 10, 20, 30, 50],
)
q = 0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=q,
    greater_is_better=False,  # maximize the negative loss
)
qrf = RandomForestQuantileRegressor(random_state=0, q=q)
search_05p = RandomizedSearchCV(
    qrf,
    param_grid,
    n_iter=20,  # increase this if computational budget allows
    scoring=neg_mean_pinball_loss_05p_scorer,
    n_jobs=2,
    random_state=0,
).fit(X_train, y_train)
pprint(search_05p.best_params_)

q = 0.95
neg_mean_pinball_loss_95p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=q,
    greater_is_better=False,  # maximize the negative loss
)
search_95p = clone(search_05p).set_params(
    estimator__q=q,
    scoring=neg_mean_pinball_loss_95p_scorer,
)
search_95p.fit(X_train, y_train)

q = 0.50
neg_mean_pinball_loss_50p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=q,
    greater_is_better=False,  # maximize the negative loss
)
search_50p = clone(search_05p).set_params(
    estimator__q=q,
    scoring=neg_mean_pinball_loss_95p_scorer,
)
search_50p.fit(X_train, y_train)
pprint(search_50p.best_params_)

joblib.dump(search_05p, "models/saved_models/search_05p.pkl")
joblib.dump(search_50p, "models/saved_models/search_50p.pkl")
joblib.dump(search_95p, "models/saved_models/search_95p.pkl")

# Pick 100 random samples from validation set
indices = np.random.choice(len(X_val), size=100, replace=False)

y_lower = search_05p.predict(X_val.iloc[indices])
y_upper = search_95p.predict(X_val.iloc[indices])

y_true = y_val.iloc[indices].values

x_axis = np.arange(len(indices))  # just use sample index

plt.figure(figsize=(10, 6))
plt.fill_between(x_axis, y_lower, y_upper, alpha=0.3, label='90% prediction interval', color='gray')
plt.plot(x_axis, y_true, 'o', color='blue', markersize=4, label='True Salary')

plt.xlabel("Sample Index (Random Subset)")
plt.ylabel("Predicted Salary")
plt.title("90% Prediction Interval on Random Subset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()