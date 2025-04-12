# The baseline model of K-Means and SC in Mnist dataset
from SEC_Model import clustering_accuracy
# load mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X[:5000] # shape (5000, 784)
y = y[:5000].astype(int) # shape (5000,)

print(X.shape, y.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=100)

X = pca.fit_transform(X)

# Split into in-sample/out-of-sample

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_train = X_train.T  # shape (784, n_train)
X_test = X_test.T    # shape (784, n_test)

X_train = np.array(X_train, dtype=np.float64)
X_test = np.array(X_test, dtype=np.float64)

print(X_train.shape, y_train.shape)

# Use K-Means as a baseline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print("shape",X_train.T.shape)
kmeans = KMeans(n_clusters=10, random_state=42).fit(X_train.T)
y_train_pred = kmeans.labels_

print("K-Means training accuracy:", clustering_accuracy(y_train, y_train_pred))

import numpy as np
from sklearn.cluster import SpectralClustering

#Use Spectral Clustering as a baseline

print("shape",X_train.T.shape)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
y_train_pred = spectral.fit_predict(X_train.T)


print("Spectral Clustering training accuracy:", clustering_accuracy(y_train, y_train_pred))