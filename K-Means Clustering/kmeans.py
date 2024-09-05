import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# load the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('customers.csv')

custer_data_1 = customer_data

customer_data = customer_data.drop(['Gender'], axis=1)

customer_data.head()
customer_data.describe()
customer_data.info()

customer_data.isnull().sum()

cust_shape = custer_data_1.shape  # Put the shape of the dataset here - 1.1 of Gradescope tests
print(cust_shape)

customer_data.isnull().sum()

matrix = customer_data.corr()
sns.heatmap(matrix, annot=True)
plt.show()

col_param_1 = "Annual Income (k$)"  # Put the name of the first column here - 1.2 of Gradescope tests
col_param_2 = "Spending Score (1-100)" # Put the name of the second column here - 1.3 of Gradescope tests

matrix_values = matrix.values.flatten()
matrix_max = np.max(matrix_values[(matrix_values < 1) & (matrix_values > 0)])
matrix_max = round(matrix_max, 2) # 1.4 of Gradescope tests

customer_data = customer_data[[col_param_1, col_param_2]] # subset the dataset for the 2 columns we are interested in

num_clust = 15
wcss_list = []

for i in range(1, num_clust + 1):
    print(f'k={i}')

    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(customer_data)  # fit the data to the kmeans instance you have defined
    wcss_list.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))

plt.plot(range(1, num_clust + 1), wcss_list, marker='o', color='red')

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()

chosen_cluster = 6 # 1.5 of Gradescope tests

kmeans =  KMeans(n_clusters=chosen_cluster, init='k-means++', n_init=10, random_state=42)
kmeans.fit(customer_data)  # fit the instance with the data

Y = kmeans.fit_predict(customer_data)

print(kmeans.cluster_centers_)  # Print the cluster centres
print(Y)

max_centre = np.max(kmeans.cluster_centers_) # 1.6 of Gradescope tests

X = customer_data.iloc[:, :].values
# plotting all the clusters and their Centroids

plt.figure(figsize=(20, 10))

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='lime', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='maroon', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], c='gold', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], c='blue', label='Cluster 5')
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], c='black', label='Cluster 6')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.legend(loc="upper right")

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.show()
