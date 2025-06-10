import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import model_selection, linear_model, metrics
import DataAnalytics.dataAnalysis

data_path = "Data/ds_salaries.csv"
save_path = "Data/figures/lin_reg"
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(data_path)

# preprocessing
one_hot = [["job_title", "employment_type", "salary_currency", "employee_residence", "company_location"]]
for col in df.columns:
    if(df[col].dtype == 'object'):
        if col in one_hot:
            df[col] = DataAnalytics.dataAnalysis.columnOneHotEncoding(df[col])
        else:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
DataAnalytics.dataAnalysis.cleanDataAndReturnSummaryStatistics(df)

# correlation matrix
# features = all columns except salary
X = df.drop(["salary", "salary_in_usd"], axis=1)
Y = df[["salary_in_usd"]]

# 80-10-10 train-val-test split
X_train, X_test_val, Y_train, Y_test_val = model_selection.train_test_split(X, Y, train_size=0.8)
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_test_val, Y_test_val, test_size=0.5)

corr_matrix = df.corr()
"""
print(corr_matrix["salary_in_usd"]):
work_year             0.228290
experience_level      0.327173
employment_type      -0.010329
job_title             0.120875
salary               -0.023676
salary_currency       0.430450
salary_in_usd         1.000000
employee_residence    0.414039
remote_ratio         -0.064171
company_location      0.405183
company_size         -0.000372

highest corr: salary_currency, employee_residence, company_location
"""

plt.scatter(X["salary_currency"],Y)
plt.title("Salary Currency vs Salary in USD")
plt.savefig(os.path.join(save_path, "curr_sal"))

plt.scatter(X["employee_residence"],Y)
plt.title("Employee Residence vs Salary in USD")
plt.savefig(os.path.join(save_path, "res_sal"))

plt.scatter(X["company_location"],Y)
plt.title("Company Location vs Salary in USD")
plt.savefig(os.path.join(save_path, "loc_sal"))

# Regression
def find_error(cols):
    model = linear_model.LinearRegression()
    model.fit(X_train[[cols]], Y_train)
    Y_train_pred = model.predict(X_train[[cols]])
    Y_val_pred = model.predict(X_val[[cols]])

    plt.scatter(X_train[cols],Y_train)
    plt.xlabel(cols)
    plt.ylabel("Salary in USD ($)")
    plt.plot(X_train[cols], Y_train_pred)
    plt.title(cols + " vs Salary in USD")
    plt.savefig(os.path.join(save_path, (cols + "_sal")))

    return [metrics.mean_squared_error(Y_train, Y_train_pred), metrics.mean_squared_error(Y_val, Y_val_pred)]

for col in ["work_year", "experience_level", "employment_type", "job_title", "salary_currency", 
            "employee_residence", "remote_ratio", "company_location", "company_size"]:
    print(col, find_error(col))

# salary_currency has lowest val error
min_val_error = "salary_currency"
model = linear_model.LinearRegression()
model.fit(X_train[[min_val_error]], Y_train)
Y_test_pred = model.predict(X_test[[min_val_error]])
print(min_val_error, "test error: ", metrics.mean_squared_error(Y_test_pred, Y_test))