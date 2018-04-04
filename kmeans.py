# import modules
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

## data setup
data = np.loadtxt("six_class.csv", delimiter=',')

## K MEANS
clustering = KMeans(n_clusters=6)
clustering.fit(data)

## get results
labels = clustering.labels_
clusters = clustering.cluster_centers_

## graph
colors = ["blue", "red", "orange", "yellow", "green", "brown"]
for pt, cluster in zip(data, labels):
    i, j = pt[0], pt[1]
    plt.scatter(i, j, color=colors[cluster % len(colors)], alpha=.5)

for n, pt in enumerate(clusters):
    i, j = pt[0], pt[1]
    plt.scatter(
        i,
        j,
        color=colors[n % len(colors)],
        marker="X",
        s=100,
        edgecolor='black',
        linewidth='3')
plt.savefig("clusters")
