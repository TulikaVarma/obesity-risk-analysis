import numpy as np


class CLARANS:
    def __init__(self, n_clusters: int, maxneighbor: int, numlocal: int):
        """
        n_clusters: Number of clusters
        maxneighbor: Maximum number of random (medoid, non-medoid) swaps in each iteration
        numlocal: Number of local searches
        """
        self.n_clusters = n_clusters
        self.maxneighbor = maxneighbor
        self.numlocal = numlocal
        self.medoids = None
        self.labels_ = None

    def fit(self, X: np.ndarray):
        """
        Performs clustering using CLARANS.
        X: Input data as a numpy array
        ut is going to return the cluster labels
        """
        best_medoids = None
        best_cost = float('inf')

        for local_search in range(self.numlocal):
            # Initializign the medoids randomly
            current_medoids_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            current_cost = self.calculate_cost(X, current_medoids_idx)

            for _ in range(self.maxneighbor):
                updated = False

                # Randomly selecting a medoid and a non-medoid point
                for medoid_idx in range(len(current_medoids_idx)):
                    for non_medoid_idx in range(X.shape[0]):
                        if non_medoid_idx in current_medoids_idx:
                            continue

                        new_medoids_idx = current_medoids_idx.copy()
                        new_medoids_idx[medoid_idx] = non_medoid_idx

                        # Calculating the cost for the new medoids
                        new_cost = self.calculate_cost(X, new_medoids_idx)

                        if new_cost < current_cost:
                            current_medoids_idx = new_medoids_idx
                            current_cost = new_cost
                            updated = True
                            break

                    if updated:
                        break

                if not updated:
                    # If no improvement is found,then we gonna stop
                    break

            if current_cost < best_cost:
                best_medoids = current_medoids_idx
                best_cost = current_cost

        self.medoids = X[best_medoids]
        self.labels_ = self.assign_clusters(X, self.medoids)
        return self.labels_

    def calculate_cost(self, X: np.ndarray, medoids_idx: np.ndarray):
        """
        Calculate the total cost 
        X: Input data
        medoids_idx: indexes of the current medoids
        """
        medoids = X[medoids_idx]
        distances = self.euclidean_distance(X, medoids)
        return np.sum(np.min(distances, axis=1))

    def assign_clusters(self, X: np.ndarray, medoids: np.ndarray):
        """
        Assigning each data point to the closest medoid.
        X: Input data
        medoids: Current medoids
        """
        distances = self.euclidean_distance(X, medoids)
        return np.argmin(distances, axis=1)

    def euclidean_distance(self, X1: np.ndarray, X2: np.ndarray):
        """
        Compute the Euclidean distance between all pairs (x, y) where x is a row in X1 and y is a row in X2.
        """
        # calculating the distance between each object in X (data points ) and each centroid and return it as matrix
        return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        """
        Compute the silhouette score for the clustering results.
        This code was taken from the silhouette in kmean 
        clustering: Cluster labels 
        X: Input data.
        """
        # silhouette = (b - a)/max(a, b) where b is distance to second nearest cluster centroid and a is to first nearest centroid
        # ie it's own cluster centroid 
        s_list = []
        for i in range(X.shape[0]):
            x = X[i].reshape(1, X.shape[1])
            label_x = clustering[i]
            same_cluster_points = X[clustering == label_x]
            a = np.mean(self.euclidean_distance(x, same_cluster_points)) if len(same_cluster_points) > 1 else 0
            b = float('inf')
            for cluster in range(self.n_clusters):
                if cluster == label_x:
                    continue
                other_cluster_points = X[clustering == cluster]
                if len(other_cluster_points) > 0:
                    b = min(b, np.mean(self.euclidean_distance(x, other_cluster_points)))
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            s_list.append(s)
        return np.mean(s_list)