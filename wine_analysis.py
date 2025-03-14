# libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine_data=pd.read_csv(r"C:\Users\Abraham\Desktop\DOINGS\WineQT.csv")

print("\ndata info")
print(wine_data.info)
print(wine_data.head)

print("\n null values")
print(wine_data.isnull().sum)



# Droping the icolomn as  its not useful for analysis
wine_data.drop(columns=['Id'], inplace=True)


# Visualizing the distribution of wine quality scores
plt.figure(figsize=(8, 5))
sns.countplot(x=wine_data["quality"], palette="viridis")
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

# Compute correlation matrix
correlation_matrix = wine_data.corr()


# Plot heatmap of feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# Drop unnecessary columns if present
wine_data.drop(columns=["Id"], errors="ignore", inplace=True)

# Plot the relationship between density and quality
plt.figure(figsize=(8, 5))
sns.boxplot(x=wine_data["quality"], y=wine_data["density"], palette="viridis")
plt.title("Density vs Wine Quality")
plt.xlabel("Quality Score")
plt.ylabel("Density")
plt.show()


sns.boxplot(x="quality", y="fixed acidity", data=wine_data, hue="quality", palette="magma", legend=False)
plt.title("Fixed Acidity vs Wine Quality")
plt.xlabel("Quality Score")
plt.ylabel("Fixed Acidity")
plt.show()



# Compute correlation of density and acidity with wine quality
correlations = wine_data[["density", "fixed acidity", "volatile acidity", "citric acid", "quality"]].corr()
correlations["quality"].sort_values(ascending=False)


# Converting quality into binary classification (Good: quality >= 6, Bad: quality < 6)
wine_data["quality_label"] = (wine_data["quality"] >= 6).astype(int)

# Define features and target variable
X = wine_data.drop(columns=["quality", "quality_label"])  # Features
y = wine_data["quality_label"]  # Target

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features for models sensitive to scale (SGD, SVC)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display shapes of train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

