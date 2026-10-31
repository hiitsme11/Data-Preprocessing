"""
Week 1 Internship Project: Student Performance Data Preprocessing
Dataset: small synthetic sample created for learning and demonstration.
Run: python preprocess.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Load the raw CSV file
DATA_FILE = "student_performance_raw.csv"
df = pd.read_csv(DATA_FILE)

print("\n--- Original dataset preview ---")
print(df.head())
print("\nOriginal shape:", df.shape)

# 2. Inspect the dataset
print("\n--- Dataset information ---")
df.info()

print("\n--- Missing values in each column ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

# 3. Remove duplicate records
df = df.drop_duplicates()

# 4. Handle missing numeric values using the median
numeric_columns = ["Study_Hours", "Attendance"]
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# 5. Handle missing categorical values, if any, using the mode
categorical_columns = df.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

# 6. Feature selection: keep useful columns for this demonstration
selected_columns = [
    "Age", "Study_Hours", "Attendance", "Previous_Score",
    "Internet_Access", "Learning_Mode", "Final_Result"
]
model_data = df[selected_columns].copy()

# 7. Encode categorical columns
model_data = pd.get_dummies(
    model_data,
    columns=["Internet_Access", "Learning_Mode"],
    drop_first=True,
    dtype=int
)

# Encode target label: Pass = 1, Fail = 0
model_data["Final_Result"] = model_data["Final_Result"].map(
    {"Pass": 1, "Fail": 0}
)

# 8. Normalize numeric input features using Min-Max scaling
scale_columns = ["Age", "Study_Hours", "Attendance", "Previous_Score"]
scaler = MinMaxScaler()
model_data[scale_columns] = scaler.fit_transform(model_data[scale_columns])

# 9. Save cleaned dataset
os.makedirs("output", exist_ok=True)
cleaned_path = os.path.join("output", "student_performance_cleaned.csv")
model_data.to_csv(cleaned_path, index=False)

# 10. Exploratory Data Analysis (EDA)
print("\n--- Descriptive statistics (raw cleaned data) ---")
print(df[["Age", "Study_Hours", "Attendance", "Previous_Score"]].describe())

print("\n--- Final preprocessed dataset preview ---")
print(model_data.head())
print("\nFinal shape:", model_data.shape)
print("\nSaved:", cleaned_path)

# Create a simple EDA chart
plt.figure(figsize=(7, 4))
df["Final_Result"].value_counts().plot(kind="bar")
plt.title("Student Final Result Counts")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(os.path.join("output", "result_distribution.png"))
plt.show()
