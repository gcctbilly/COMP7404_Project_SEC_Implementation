# The baseline model of K-Means and SC in Optdigits dataset
from SEC_Model import clustering_accuracy
# Optdigits Dataset
# load data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()

X_full = digits.data  # shape (1797, 64)
y_full = digits.target  # shape (1797,) with classes [0..9]

# We want X in shape (d, n), so transpose
X_full = X_full.T  # now shape is (64, 1797)

#of clusters = 10
c = 10

# 2. Split into in-sample/out-of-sample
# The paper often uses "seen" vs "unseen" terminology.
# We'll do 60% in-sampl, 40% out-of-sample to illustrate.
X_train, X_test, y_train, y_test = train_test_split(
    X_full.T, 
    y_full, 
    test_size=0.4,
    random_state=42,
    stratify=y_full
)
X_train = X_train.T  # shape (64, n_train)
X_test = X_test.T    # shape (64, n_test)

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