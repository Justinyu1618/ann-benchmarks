from __future__ import absolute_import
from flannel import AnnoyIndex as FlannelIndex
from ann_benchmarks.algorithms.base import BaseANN
import csv, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

word_freqs = pickle.load(open("./data/word_freqs.pkl", "rb"))
glove_labels = pickle.load(open("./data/glove_labels.pkl", "rb"))

class Flannel(BaseANN):
    def __init__(self, metric, n_trees, top_p, n_jobs, model_top_p):
        self._n_trees = n_trees
        self._search_k = None
        self._clusters_p = None
        self._metric = metric
        # self._n_clusters = n_clusters
        self._top_p = top_p
        # self._with_neighbors = with_neighbors
        self._n_jobs = n_jobs
        self.model_top_p = model_top_p

    def fit(self, X):
        self._annoy = FlannelIndex(X.shape[1], metric=self._metric)
        weights = []
        for i, x in enumerate(X):
            # print("adding item: ", i)
            label = glove_labels[i]
            freq = word_freqs[label] if label in word_freqs else 1
            weights.append(freq)
            # print(freq)

            self._annoy.add_item(i, x.tolist(), freq)
        weights = np.array(weights)
        print("building")
        print(self._n_trees, self._top_p, self._n_jobs)
        # self._annoy.build(self._n_trees, self._n_clusters, self._top_p, self._with_neighbors, self._n_neighbors)
        self._annoy.build(self._n_trees, self._top_p, self._n_jobs)
        print("built")

        # other_top_p = 0.5
        is_in = weights > np.percentile(weights, 100 - self.model_top_p * 100)

        model = LogisticRegression()
        model.fit(X, is_in)

        self._annoy.set_model(model.coef_[0].tolist(), model.intercept_[0])


    def set_query_arguments(self, search_k):
        self._search_k = search_k
        # self._clusters_p = clusters_p

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)

    def __str__(self):
        # return 'Flannel(n_trees=%d, search_k=%d, n_clusters=%d, top_p=%d, with_neighbors=%d, n_neighbors=%d)' % (self._n_trees,
                                                #    self._search_k, self._n_clusters, self._top_p, self._with_neighbors, self._n_neighbors)
        return 'Flannel(n_trees=%d, search_k=%d, top_p=%d)' % (self._n_trees,
                                                   self._search_k, self._top_p)
