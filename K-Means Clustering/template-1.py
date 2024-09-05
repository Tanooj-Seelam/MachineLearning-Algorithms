# Load the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# load the data from csv file to a Pandas DataFrame

customer_data = # YOUR CODE HERE

# Whenever dealing with any kind of data, try to see what the head values look like and a description of the data, using - head
customer_data.head()

# Also practice describing the data as you get valuble insights from it, using - describe methods
customer_data.describe()

# Additionally you may look at the shape and information about the data for better understanding of what you are dealing with, using - info
customer_data.info()

# Perform a Null check for the data

customer_data.isnull().sum()

# What is the shape of the data?

cust_shape = # YOUR CODE HERE - Put the shape of the dataset here - 1.1 of Gradescope tests
print(cust_shape)

# Always do a null check as it gives us an idea of whether or not we need to impute or drop any sample(s)
customer_data.isnull().sum()

# Use the correlation matrix to find which two columns are most relevant to clustering customers - plot the heatmap to get a better view of values
matrix = # YOUR CODE HERE

sns.heatmap(matrix, annot = True) # This plots the heatmap

plt.show() # This generates the plot for us to see

# Which are the 2 columns that should be most relevant to us i.e. least correlated

col_param_1 = # Put the name of the first column here - 1.2 of Gradescope tests
col_param_2 = # Put the name of the first column here - 1.3 of Gradescope tests

# What is the max positive value present in the matrix that is not equal to 1
matrix_max = # 1.4 of Gradescope tests

# From the above plot you shall notice: Choosing the Annual Income & Spending Score columns would be the best to serve our purpose

customer_data = # YOUR CODE HERE - subset the dataset for the 2 columns we are interested in

# Use the Elbow method to find the best number of culsters to use for the data we have

num_clust = 15 # Initialize with a different value to play around with the results if you so wish

wcss_list = [] # WCSS is an abbreviation for Within Cluster Sum of Squares. It measures how similar the points within a cluster are using variance as the metric

for i in range(1, num_clust + 1):
    
    print(f'k={i}') # This helps to display which cluster number is being trained now
    
    kmeans = # write a KMeans instance with the following prescribed parameters - n_clusters = i (since we shall iterate over the multiple values of clusters we want to try out in the loop as defined above), init = 'k-means++', n_init = 10 and random_state = 42
    kmeans.fit(customer_data) # fit the data to the kmeans instance you have defined
    
    # The inertia_ method returns wcss for that model as can be seen below
    
    wcss_list.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))

plt.plot(range(1, num_clust + 1), wcss_list, marker = 'o', color = 'red')

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()

# Choose the best approximate cluster value using the Elbow method above - for our case we would notice that 6 would be an ideal value, but you may try out with different values

chosen_cluster = # 1.5 of Gradescope tests

kmeans = # write a KMeans instance with the following prescribed parameters - n_clusters = the chosen cluster value as reported above, init = 'k-means++', n_init = 10 and random_state = 42.fit(customer_data)
kmeans.fit(customer_data) #fit the instance with the data

Y = kmeans.fit_predict(customer_data)

print(kmeans.cluster_centers_) # Print the cluster centres
print(Y)

# Report the max value of the cluster centres that you see from the output matrix above

max_centre = # 1.6 of Gradescope tests

# This last part will show you how to visualize Kmeans Clusters so to get a clear understanding of what exactly it is, that you have done
# and also see for yourself if your clusters seem to be enough and contiguous

X = customer_data.iloc[:, :].values
# plotting all the clusters and their Centroids

plt.figure(figsize = (20, 10))

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c = 'lime', label = 'Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'maroon', label = 'Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], c = 'gold', label = 'Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], c = 'violet', label = 'Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], c = 'blue', label = 'Cluster 5')
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], c = 'black', label = 'Cluster 6')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'cyan', label = 'Centroids')

plt.legend(loc = "upper right")

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.show()