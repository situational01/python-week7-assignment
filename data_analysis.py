# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
try:
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

# Show first 5 rows
print("First 5 rows:")
print(data.head())

# Check data types and missing values
print("\nData info:")
print(data.info())

print("\nMissing values check:")
print(data.isnull().sum())

# No missing data in this dataset, so we don't need to clean anything

# Basic statistics
print("\nBasic statistics:")
print(data.describe())

# Group by species and find mean
print("\nAverage values by species:")
print(data.groupby('species').mean())

# ---- Visualizations ----
sns.set()  # use seaborn style for all plots

# Line Chart - petal length sorted
sorted_data = data.sort_values('petal length (cm)')
plt.figure(figsize=(8, 5))
plt.plot(sorted_data['petal length (cm)'].values)
plt.title("Line Chart: Petal Length (Sorted)")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Bar Chart - average sepal width by species
plt.figure(figsize=(7, 5))
avg_sepal = data.groupby('species')['sepal width (cm)'].mean()
avg_sepal.plot(kind='bar', color='skyblue')
plt.title("Bar Chart: Average Sepal Width")
plt.xlabel("Species")
plt.ylabel("Average Sepal Width (cm)")
plt.tight_layout()
plt.show()

# Histogram - petal length
plt.figure(figsize=(7, 5))
plt.hist(data['petal length (cm)'], bins=15, color='orange', edgecolor='black')
plt.title("Histogram: Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Scatter Plot - sepal length vs petal length
plt.figure(figsize=(7, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=data)
plt.title("Scatter Plot: Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# ---- Observations ----
print("\nObservations:")
print("- Iris-setosa has smaller petals compared to the other species.")
print("- There seems to be a relationship between sepal length and petal length.")
print("- The histogram shows that petal lengths are not evenly distributed.")
