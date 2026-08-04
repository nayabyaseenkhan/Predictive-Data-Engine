import pandas as pd

# Load the dataset
df = pd.read_csv("data/train.csv")

# Display first 5 rows
print("\n===== First 5 Rows =====")
print(df.head())

# Dataset shape
print("\n===== Shape =====")
print(df.shape)

# Column names
print("\n===== Columns =====")
print(df.columns)

# Data types
print("\n===== Data Types =====")
print(df.dtypes)

# Dataset information
print("\n===== Dataset Info =====")
print(df.info())

# Statistical summary
print("\n===== Statistical Summary =====")
print(df.describe())

# Missing values
print("\n===== Missing Values =====")
print(df.isnull().sum())