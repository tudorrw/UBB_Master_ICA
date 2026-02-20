import numpy as np
from collections import Counter


def euclidean_distance(p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) **2))


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict_proba(self, X_test):
        X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)
        X_train_sq = np.sum(self.X_train**2, axis=1)
        dot_product = np.dot(X_test, self.X_train.T)
        dists = np.sqrt(np.maximum(X_test_sq - 2 * dot_product + X_train_sq, 0))
        
        probs = []
        for i in range(dists.shape[0]):
            k_indices = np.argsort(dists[i])[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prob_positive = np.sum(k_nearest_labels == 1) / self.k
            probs.append(prob_positive)
            
        return np.array(probs)
    
