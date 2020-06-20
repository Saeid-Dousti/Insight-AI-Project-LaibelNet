import numpy as np
import streamlit as st
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class imageset_cluster():

    def __init__(self, features, number_clstrs=None, min_clustr=2, max_clustr=10):

        self.features = features
        self.number_clstrs = number_clstrs
        self.min_clustr = min_clustr
        self.max_clustr = max_clustr

        self.kmns_num_clstrs = self.number_clstrs
        self.kmns = []
        self.kmns_clstrs = []
        self.kmns_silhout = []

        self.gmm_num_clstrs = self.number_clstrs
        self.gmm = []
        self.gmm_clstrs = []
        self.gmm_silhout = []

        self.kmns_silhout_range = []
        self.gmm_silhout_range = []

        if self.number_clstrs is None:
            self.clustering2unknown()
        else:
            self.clustering2known()

    def clustering2known(self):

        self.kmns = KMeans(n_clusters=self.number_clstrs, random_state=0).fit(self.features)
        self.kmns_clstrs = self.kmns.labels_
        self.kmns_silhout = silhouette_score(self.features, self.kmns_clstrs)

        self.gmm = GMM(n_components=self.number_clstrs, random_state=0).fit(self.features)
        self.gmm_clstrs = self.gmm.predict(self.features)
        self.gmm_silhout = silhouette_score(self.features, self.gmm_clstrs)

    def clustering2unknown(self):
        silhout = dict()
        opt_clustr = dict()
        optimized_model = dict()

        model_per_cluster = [KMeans(n_clusters=k, random_state=0).fit(self.features) for k in range(self.min_clustr, self.max_clustr)]

        silhout['KMeans'] = [silhouette_score(self.features, k_cluster.labels_) for k_cluster in model_per_cluster]
        opt_clustr['KMeans'] = np.argmax(silhout['KMeans']) + self.min_clustr
        optimized_model['KMeans'] = KMeans(n_clusters=opt_clustr['KMeans'], random_state=0).fit(self.features)

        self.kmns_num_clstrs = opt_clustr['KMeans']
        self.kmns = optimized_model['KMeans']
        self.kmns_clstrs = self.kmns.labels_
        self.kmns_silhout = silhouette_score(self.features, self.kmns_clstrs)
        self.kmns_silhout_range = silhout['KMeans']

        # --------------------
        model_per_cluster = [GMM(n_components=k, random_state=0).fit(self.features) for k in range(self.min_clustr, self.max_clustr)]

        silhout['GMM'] = [silhouette_score(self.features, k_cluster.predict(self.features)) for k_cluster in model_per_cluster]
        opt_clustr['GMM'] = np.argmax(silhout['GMM']) + self.min_clustr
        optimized_model['GMM'] = GMM(n_components=opt_clustr['GMM'], random_state=0).fit(self.features)

        self.gmm_num_clstrs = opt_clustr['GMM']
        self.gmm = optimized_model['GMM']
        self.gmm_clstrs = self.gmm.predict(self.features)
        self.gmm_silhout = silhouette_score(self.features, self.gmm_clstrs)
        self.gmm_silhout_range = silhout['GMM']