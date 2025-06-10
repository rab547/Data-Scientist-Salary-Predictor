import os
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import neighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataAnalytics')))
from dataAnalysis import *

data_path = "Data/ds_salaries.csv"
save_path = "Data/figures/knn"
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(data_path)

# preprocessing
for col in df.columns:
    if(df[col].dtype == 'object'):
        """
        if col in one_hot:
            df[col] = dataAnalysis.columnOneHotEncoding(df[col])
        else:
        """
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
cleanDataAndReturnSummaryStatistics(df)

X = df.drop(["salary_in_usd"], axis=1)
Y = df["salary_in_usd"]
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
print("3-NN")
print(f"train accuracy: {knn.score(train_x, train_y)}")
print(f"test accuracy: {knn.score(test_x, test_y)}")

# KNN with multiple k values
train_acc = []
test_acc = []
k_values = range(1, 51)

for k in k_values:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(train_x, train_y)

    train_acc.append(knn.score(train_x, train_y))
    test_acc.append(knn.score(test_x, test_y))
    """
    print(f"k = {k + 1}")
    print(f"train accuracy: {knn.score(train_x, train_y)}")
    print(f"test accuracy: {knn.score(test_x, test_y)}")
    """

plt.scatter(k_values, train_acc, label="Train Accuracies", s = 10)
plt.scatter(k_values, test_acc, label="Test Accuracies", s = 10)
plt.title("KNN K-Neighbors vs Accuracy")
plt.xlabel("K-Neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# KNN with multiple sets of parameters
train_acc = []
test_acc = []
n_values = range(1, len(df.columns))

for n in n_values:
    knn = neighbors.KNeighborsClassifier(weights='uniform')
    knn.fit(train_x.iloc[:,:n], train_y)

    train_acc.append(knn.score(train_x.iloc[:,:n], train_y))
    test_acc.append(knn.score(test_x.iloc[:,:n], test_y))

plt.scatter(n_values, train_acc, label="Train Accuracies", s = 10)
plt.scatter(n_values, test_acc, label="Test Accuracies", s = 10)
plt.title("KNN Number of Features vs Accuracy")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# KNN with normalized data
train_acc = []
test_acc = []
s_values = np.linspace(-5, 5)
train_x_norm = (train_x - train_x.mean()) / train_x.std()
test_x_norm = (test_x - test_x.mean()) / test_x.std()

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x_norm, train_y)
print("Normalized 3-NN")
print(f"train accuracy: {knn.score(train_x_norm, train_y)}")
print(f"test accuracy: {knn.score(test_x_norm, test_y)}")

for s in s_values:
    train_x_scaled = train_x_norm * pow(10, s)
    test_x_scaled = test_x_norm * pow(10, s)

    knn = neighbors.KNeighborsClassifier(weights='uniform')
    knn.fit(train_x_scaled, train_y)

    train_acc.append(knn.score(train_x_scaled, train_y))
    test_acc.append(knn.score(test_x_scaled, test_y))

plt.scatter(s_values, train_acc, label="Train Accuracies", s = 10)
plt.scatter(s_values, test_acc, label="Test Accuracies", s = 10)
plt.title("KNN Input Scale vs Accuracy")
plt.xlabel("Input Scale")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()