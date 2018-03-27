import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class KMeans:

    def __init__(self, X, n_clusters, max_iter=float('Inf')):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.X = X
        self.n, self.d = self.X.shape

    def initialize_random_centroids(self):
        random_indices = [np.random.randint(0, self.n - 1) for _ in range(self.n_clusters)]
        centroids = self.X[random_indices, :]
        self.centroids = np.array(centroids)

    def compute_average(self):
        means = []
        for k in range(self.n_clusters):
            members_indices = np.argwhere(self.memberships == k).reshape(-1)
            members_k = self.X[members_indices, :]
            members_k_mean = np.average(members_k, axis=0)
            means.append(members_k_mean.tolist())
        self.centroids = np.array(means)

    def assign_clusters(self):
        distances = euclidean_distances(self.X, self.centroids)
        self.memberships = np.argmin(distances, axis=1)

    def mean_distances_to_centroids(self):
        sum = 0
        for k in range(self.n_clusters):
            centroid_k = self.centroids[k, :].reshape(1, -1)
            members_indices = np.argwhere(self.memberships == k).reshape(-1)
            members_k = self.X[members_indices, :]
            if members_k.size > 0:
                distances_k = euclidean_distances(members_k, centroid_k)
            sum += np.sum(distances_k)
        return sum / self.n

    def fit(self):
        iter = 0
        self.initialize_random_centroids()
        prev_mean_distance = float('Inf')
        current_distance = 0.0
        while abs(prev_mean_distance - current_distance) > 0.0001 and iter < self.max_iter:
            self.assign_clusters()
            self.compute_average()
            prev_mean_distance = current_distance
            current_distance = self.mean_distances_to_centroids()
            iter += 1
        return current_distance * self.n

if __name__ == "__main__":
    data = 'data/data_2_large'
    X = np.loadtxt(data + '.txt')
    kmeans = KMeans(X, 3)
    kmeans.fit()
    print(kmeans.memberships)


# TODO:
# test this