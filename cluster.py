import numpy as np
import streamlit as st
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class imageset_cluster():

    def __init__(self, features, number_clstrs=None, min_clustr=2, max_clustr=10):

        self.number_clstrs = number_clstrs
        self.kmns = []
        self.kmns_clstrs = []

        self.gmm = []
        self.gmm_clstrs = []
        self.gmm_silhout = []

        if self.number_clstrs is None:
            self.clustering2UNknown(features, min_clustr, max_clustr)
        else:
            self.clustering2known(features)


    def clustering2known(self, features):
        self.kmns = KMeans(n_clusters=self.number_clstrs, random_state=0).fit(features)
        self.kmns_clstrs = self.kmns.labels_
        self.kmns_silhout = silhouette_score(features, self.kmns_clstrs)

        self.gmm = GMM(n_components=self.number_clstrs, random_state=0).fit(features)
        self.gmm_clstrs = self.gmm.predict(features)
        self.gmm_silhout = silhouette_score(features, self.gmm_clstrs)


    def clustering2UNknown(self, features, min_clustr, max_clustr):
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
