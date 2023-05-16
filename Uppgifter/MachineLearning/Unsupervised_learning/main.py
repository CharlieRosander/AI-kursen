import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the whole dataset
home_data = pd.read_csv('./Docs/housing.csv')

# Load the dataset with only the longitude, latitude and median house value
home_data_median = pd.read_csv('./Docs/housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
home_data_median.head()

# Load the dataset with only the longitude, latitude and ocean proximity

home_data_ocean = pd.read_csv('./Docs/housing.csv', usecols = ['longitude', 'latitude', 'ocean_proximity'])
home_data_ocean.head()

# Convert ocean proximity to numeric values 
home_data_ocean['ocean_proximity'] = home_data_ocean['ocean_proximity'].map({'<1H OCEAN': 1, 'INLAND': 2, 'NEAR OCEAN': 3, 'NEAR BAY': 4, 'ISLAND': 5})
home_data_ocean.head()

# Visualize the data
plt.figure(figsize = (10, 8))
sns.scatterplot(data = home_data_median, x = 'longitude', y = 'latitude', hue = 'median_house_value')

# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(home_data_median[['latitude', 'longitude']], home_data_median[['median_house_value']], test_size=0.33, random_state=0)

# Normalize the data

# Print before normalization
print(f"Before normalization:\n{X_train[0:5]}\n")

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# Print after normalization
print(f"After normalization:\n{X_train_norm[0:5]}")

# Fitting and Evaluating the Model

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

# Visualize the clusters
plt.figure(figsize = (10, 8))
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)

# Evaluate the model using Silhouette Score

silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

# Finding the optimal number of clusters using a for loop and Silhouette Score

K = range(2, 9)
fits = []
score = []


for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
    
    # append the model to fits
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))


# Visualize the silhouette scores
plt.figure(figsize = (10, 8))
plt.plot(K, score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for k clusters')
plt.show()


# We see that the optimal number of clusters seems to be 5, as the silhouette score is highest for k = 5 without overfitting, which k = 7 and k = 8 seems to do.

# Visualizing the clusters with respect to the median house value in a boxplot
sns.boxplot(x = fits[3].labels_, y = y_train['median_house_value'])

# Implementing the model on the test data with the new k value of 5

kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
kmeans.fit(X_test_norm)

# Visualize the clusters
plt.figure(figsize = (10, 8))
sns.scatterplot(data = X_test, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.show()

# Visualizing the clusters with respect to the median house value in a boxplot with the same colors as the scatterplot
sns.boxplot(x = kmeans.labels_, y = y_test['median_house_value'])