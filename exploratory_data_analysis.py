# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
jj = pd.read_csv('jj_translated.csv')

# Basic EDA: Distribution Plots
plt.figure(figsize=(15, 10))

# Distribution plots for key numerical variables
plt.subplot(2, 3, 1)
sns.histplot(jj['Price'], bins=20, kde=True, color='blue')
plt.title('Price Distribution')

plt.subplot(2, 3, 2)
sns.histplot(jj['Cost'], bins=20, kde=True, color='green')
plt.title('Cost Distribution')

plt.subplot(2, 3, 3)
sns.histplot(jj['Commision'], bins=20, kde=True, color='purple')
plt.title('Commission Distribution')

plt.subplot(2, 3, 4)
sns.histplot(jj['Total VAT'], bins=20, kde=True, color='red')
plt.title('Total VAT Distribution')

plt.subplot(2, 3, 5)
sns.histplot(jj['Total Fee'], bins=20, kde=True, color='orange')
plt.title('Total Fee Distribution')

plt.subplot(2, 3, 6)
sns.histplot(jj['Revenue'], bins=20, kde=True, color='brown')
plt.title('Revenue Distribution')

plt.tight_layout()
plt.show()

numeric_columns = jj.select_dtypes(include=['float64', 'int64'])

# Correlation analysis for numerical variables
correlation_matrix = numeric_columns.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


