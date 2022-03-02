import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt

x,y = make_blobs(n_samples=1000,n_features=4,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.4],random_state=10)

k_means = KMeans(n_clusters=3, random_state=10)

k_means.fit(x)

y_predict = k_means.predict(x)
plt.scatter(x[:,0],x[:,1],c=y_predict)
plt.show()
print(k_means.predict((x[:30,:])))
print(metrics.calinski_harabasz_score(x,y_predict))
print(k_means.cluster_centers_)
print(k_means.inertia_)
print(metrics.silhouette_score(x,y_predict))
