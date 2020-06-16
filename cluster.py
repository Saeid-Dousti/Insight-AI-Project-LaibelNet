import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clustering2known(features, k):

    KMeans(n_clusters=k, random_state=0).fit(features)

    GMM(n_components=k, random_state=0).fit(features)


def clustering2UNknown(features, min_clustr, max_clustr):
    silhout = dict()
    opt_clustr = dict()
    optimized_model = dict()

    model_per_cluster = [KMeans(n_clusters=k, random_state=0).fit(features) for k in range(min_clustr, max_clustr)]

    silhout['KMeans'] = [silhouette_score(features, k_cluster.labels_) for k_cluster in model_per_cluster]
    opt_clustr['KMeans'] = np.argmax(silhout['KMeans']) + min_clustr
    optimized_model['KMeans'] = KMeans(n_clusters=opt_clustr['KMeans'], random_state=0).fit(features)

    model_per_cluster = [GMM(n_components=k, random_state=0).fit(features) for k in range(min_clustr, max_clustr)]

    silhout['GMM'] = [silhouette_score(features, k_cluster.predict(features)) for k_cluster in model_per_cluster]
    opt_clustr['GMM'] = np.argmax(silhout['GMM']) + min_clustr
    optimized_model['GMM'] = GMM(n_components=opt_clustr['GMM'], random_state=0).fit(features)

    return silhout, opt_clustr, optimized_model