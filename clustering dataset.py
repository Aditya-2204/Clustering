from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x = [19,75,42,88,61,33,12,56,27,94]
y = [63,29,51,76,15,84,37,48,66,23]

x = np.reshape(x, (-1,1))
y = np.reshape(y, (-1,1))

kmeans = KMeans(n_clusters=4)
kmeans.fit(x,y)

plt.scatter(x,y,c=kmeans.labels_)
plt.title("Clustering with KMeans algorithm")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()