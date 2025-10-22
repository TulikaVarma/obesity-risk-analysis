import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print(dataset.head(3))

# 1. Perform basic EDA to understand the structure and distribution of the dataset.

# Shape
print(f"Rows x Columns: {dataset.shape[0]} x {dataset.shape[1]}")

# Column groups
print("\nColumn Groups:")
numerical_col = dataset.select_dtypes(include="number").columns.tolist()
categorical_col = dataset.select_dtypes(exclude="number").columns.tolist()
print(f"Numerical ({len(numerical_col)}): {numerical_col}")
print(f"Categorical ({len(categorical_col)}): {categorical_col}")

# Data Types
print("\nData Types:\n", dataset.dtypes)

# Non-Unique Values
print("\nNonunique values:\n", dataset.nunique())

# Missing Values
print("\nMissing Values per Column:\n", dataset.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", dataset.duplicated().sum())

# Numerical Statistics
print("\nNumerical Statistics:\n", dataset[numerical_col].describe().T.round(3).to_string())