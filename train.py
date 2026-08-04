import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

df = df.drop(columns=["Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Encode the 'Sex' column
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Encode the 'Embarked' column
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})
print("\nEncoded Dataset")
print(df.head())

# Select input features
X = df.drop(columns=["PassengerId", "Name", "Ticket", "Survived"])

# Target variable
y = df["Survived"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nSelected Features:")
print(X.head())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== Dataset Split =====")
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\n Model trained successfully!")