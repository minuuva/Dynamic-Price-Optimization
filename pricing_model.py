# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

jj = pd.read_csv('jj_translate_1.0.csv')
jj.head()

# %%
# Step 1: Extract relevant features
# Here, we assume 'Order', 'Price', and 'Cost' are columns in your data
jj['Sales Frequency'] = jj.groupby('Order')['Order'].transform('count')
features = jj[['Price', 'Cost', 'Sales Frequency']]

# %%
# Step 2: Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# %%
# Step 3: Apply K-Means Clustering
# Find the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)

# %%
# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# %%
optimal_k = 5

# Apply K-Means Clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
jj['Cluster'] = kmeans_optimal.fit_predict(normalized_features)

# Display the cluster assignments
print(jj[['Order', 'Price', 'Cost', 'Sales Frequency', 'Cluster']].head())


# %%
# Analyze each cluster to see the characteristics
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Summary:")
    cluster_data = jj[jj['Cluster'] == cluster]
    print(cluster_data[['Order', 'Price', 'Cost', 'Sales Frequency']].describe())

# %%
from sklearn.linear_model import LinearRegression

# Prepare to build a model for each cluster
pricing_models = {}

# Iterate over each cluster and build a model
for cluster in range(optimal_k):
    cluster_data = jj[jj['Cluster'] == cluster]
    
    # Define features and target
    X = cluster_data[['Cost', 'Sales Frequency']].values  # Use relevant features
    y = cluster_data['Price'].values
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the model
    pricing_models[cluster] = model
    print(f"Model for Cluster {cluster} trained.")

# %%
# Predict optimal prices for each cluster
for cluster in range(optimal_k):
    cluster_data = jj[jj['Cluster'] == cluster]
    model = pricing_models[cluster]
    
    # Predict the price using the model
    cluster_data['Predicted Price'] = model.predict(cluster_data[['Cost', 'Sales Frequency']].values)
    
    # Display the predicted prices
    print(f"\nPredicted Prices for Cluster {cluster}:")
    print(cluster_data[['Order', 'Cost', 'Sales Frequency', 'Predicted Price']].head())


