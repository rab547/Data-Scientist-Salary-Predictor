import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Data/ds_salaries.csv')

features = ['work_year', 'experience_level', 'employment_type', 'job_title', 'remote_ratio', 'company_location', 'company_size']
X = df[features]
y = df['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cat_features = ['experience_level', 'employment_type', 'job_title', 'company_location', 'company_size']
num_features = ['work_year', 'remote_ratio']

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[cat_features])
X_test_cat = encoder.transform(X_test[cat_features])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

X_train_processed = np.hstack((X_train_num, X_train_cat))
X_test_processed = np.hstack((X_test_num, X_test_cat))

# Train SVR model
svr = SVR(kernel='rbf', C=150000, epsilon=20000, gamma=0.095)
svr.fit(X_train_processed, y_train)

# Make predictions
y_pred = svr.predict(X_test_processed)




# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Salary (USD)')
plt.ylabel('Predicted Salary (USD)')
plt.title('SVR: Actual vs Predicted Salaries')
plt.tight_layout()
plt.show()

