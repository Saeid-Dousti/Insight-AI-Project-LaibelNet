import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster(features, max_clustr):
    silhout = dict()
    opt_clustr = dict()

    for model in [KMeans, GMM]:
        model_per_cluster = [model(n_clusters=k, random_state=0).fit(features) for k in range(1, max_clustr)]

        silhout[model] = [[k_cluster, silhouette_score(features, k_cluster.labels_)] for k_cluster in model_per_cluster]
        opt_clustr[model] = np.argmax(silhout[model][1]) + 1


    return silhout, opt_clustr