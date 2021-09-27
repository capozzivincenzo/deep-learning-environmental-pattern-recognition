import numpy as np
from sklearn.cluster import KMeans
from keras.datasets import mnist
import metrics
import sys

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train)
sys.exit()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)

# 10 clusters
n_clusters = len(np.unique(y))

# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)

# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)

print("-----------------------")

# Evaluate the K-Means clustering accuracy.
metrics.acc(y, y_pred_kmeans)
