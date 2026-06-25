import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders
os.makedirs("charts", exist_ok=True)
os.makedirs("report", exist_ok=True)

# Column names for UCI Heart Disease Dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Load dataset
df = pd.read_csv("data/heart.csv", names=columns)

print("=" * 50)
print("HEART DISEASE DATASET ANALYSIS")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Records Found: {duplicates}")

df.drop_duplicates(inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_heart.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ---------------- CHART 1 ----------------
plt.figure(figsize=(8,5))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("charts/chart1.png")
plt.close()

# ---------------- CHART 2 ----------------
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("charts/chart2.png")
plt.close()

# ---------------- CHART 3 ----------------
plt.figure(figsize=(8,5))
sns.boxplot(x=df["chol"])
plt.title("Cholesterol Distribution")
plt.savefig("charts/chart3.png")
plt.close()

# ---------------- CHART 4 ----------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("charts/chart4.png")
plt.close()

# ---------------- CHART 5 ----------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="age",
    y="chol",
    hue="target",
    data=df
)
plt.title("Age vs Cholesterol")
plt.savefig("charts/chart5.png")
plt.close()

print("All charts generated successfully!")

# ---------------- REPORT ----------------
report = """
DATA CLEANING SUMMARY
---------------------
1. Checked for missing values.
2. Removed duplicate records.
3. Saved cleaned dataset.

KEY FINDINGS
------------
1. Heart disease cases and non-cases were identified.
2. Most patients belong to middle and older age groups.
3. Cholesterol levels vary among patients.
4. Correlation analysis revealed relationships among attributes.
5. Age and cholesterol show noticeable patterns.

CONCLUSION
----------
The dataset was successfully cleaned and analyzed.
Visualizations helped identify trends and relationships
associated with heart disease.
"""

with open("report/findings.txt", "w") as file:
    file.write(report)

print("Report generated successfully!")

print("\nPROJECT COMPLETED SUCCESSFULLY!")