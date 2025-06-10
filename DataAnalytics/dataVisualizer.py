import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import dataAnalysis


custom_colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_theme(style="whitegrid", font_scale=1.2, palette=custom_colors)

save_dir ="Data/figures"
os.makedirs(save_dir, exist_ok=True)

df = dataAnalysis.loadDataFrame("Data/ds_salaries.csv")
df = dataAnalysis.replaceNullsWithMedian(df)

numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

#create histograms for numerical columns
for col in numerical_cols:
    sns.histplot(df[col], bins=10, kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, f"Histogram of {col}.png"))
    plt.close()

#create bar plots for categorical columns
for col in categorical_cols:
    sns.barplot(df[col])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Bar Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, f"Bar plot of {col}.png"))
    plt.close()

    #randomly choosing pairs of columns
    num_pairs = np.random.choice(numerical_cols.to_list(), size=4, replace=False).reshape(2, 2)
    selected_num_cols = random.sample(numerical_cols.tolist(), 3)
    selected_cat_cols = random.sample(categorical_cols.tolist(), 3)
    num_col_pairs = list(zip(selected_cat_cols, selected_num_cols))


    #creating violin plots for numerical column + categorical column pairs
    for col1, col2 in num_col_pairs:
        sns.violinplot(x=df[col1], y=df[col2])

        plt.title(f"Violin plot of {col1} and {col2}")
        plt.xlabel("Columns")
        plt.ylabel("Values")

    plt.savefig(os.path.join(save_dir, f"Violin plot of {col1} and {col2}.png"))

#creating scatter plots for numerical column pairs

for col1, col2 in num_pairs:
    sns.scatterplot(x=df[col1], y=df[col2])

    plt.title(f"Scatter plot of {col1} and {col2}")
    plt.xlabel("Columns")
    plt.ylabel("Values")

    plt.savefig(os.path.join(save_dir, f"Scatter plot of {col1} and {col2}.png"))

#one hot encoding every categorical column

chosen_cat_cols = ["experience_level", "company_size"]
encoded_df = df[numerical_cols].copy()
for col in chosen_cat_cols:
    encoded_col = dataAnalysis.columnOneHotEncoding(df[col], df[col].unique())
    encoded_df = pd.concat([encoded_df, encoded_col], axis=1)

#computing and filtering correlation matrix
corr_matrix = encoded_df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(save_dir, f"Filtered Correlation Heatmap of Data Science Salaries (Only Strong Correlations)"))

#explore salary and job title with box plot
sns.boxplot(
    x=df["remote_ratio"],
    y=df["salary_in_usd"],
    hue=df["remote_ratio"],
    palette="coolwarm",
    legend=False
)
plt.xticks(rotation=45)
plt.title("Salary Distribution by Job Title")
plt.xlabel("Job Title")
plt.ylabel("Salary ($)")
plt.savefig(os.path.join(save_dir, f"Box plot of Job Title and Salary.png"))

#explore salary and job title with violin plot
sns.violinplot(
    x=df["remote_ratio"],
    y=df["salary_in_usd"],
    hue=df["remote_ratio"],
    palette="coolwarm",
    legend=False
)
plt.xticks(rotation=45)
plt.title("Salary Distribution by Job Title")
plt.xlabel("Job Title")
plt.ylabel("Salary ($)")
plt.savefig(os.path.join(save_dir, f"Violin plot of Job Title and Salary.png"))
