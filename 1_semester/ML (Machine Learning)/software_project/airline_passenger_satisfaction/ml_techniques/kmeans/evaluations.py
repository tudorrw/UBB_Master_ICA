import numpy as np
from math import comb
from sklearn.metrics import (v_measure_score, homogeneity_score, completeness_score,
                            silhouette_score, davies_bouldin_score, calinski_harabasz_score)


class Evaluations:
    
    def __init__(self, pred):
        self.pred = pred
    
    
    
class ExternalEvaluations(Evaluations):
    
    def __init__(self, pred, y_test):
        super().__init__(pred)
        self.y_test = y_test
    
    #calculated metrics
    def __entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs))

    def __conditional_entropy(self, labels_true, labels_pred):
        unique_labels_true, unique_labels_pred = np.unique(labels_true), np.unique(labels_pred)
        conditional_entropy = 0.0
        for c in unique_labels_pred:
            sub_labels_true = labels_true[labels_pred == c]
            conditional_entropy += (len(sub_labels_true) / len(labels_true)) * self.__entropy(sub_labels_true)
        return conditional_entropy
    
    def __entropies(self, labels_true, labels_pred):
        H_C = self.__entropy(labels_pred)
        H_K = self.__entropy(labels_true)
        H_K_given_C = self.__conditional_entropy(labels_true, labels_pred)
        H_C_given_K = self.__conditional_entropy(labels_pred, labels_true) 
        return H_C, H_K, H_K_given_C, H_C_given_K 
        
    def homogeneity(self):
        H_C, H_K, H_K_given_C, H_C_given_K = self.__entropies(self.y_test, self.pred)
        return 1 - H_K_given_C / H_K if H_K != 0 else 1.0
        
    def completeness(self):
        H_C, H_K, H_K_given_C, H_C_given_K = self.__entropies(self.y_test, self.pred)
        return 1 - H_C_given_K / H_C if H_C != 0 else 1.0
    
    def v_measure(self):
        H_C, H_K, H_K_given_C, H_C_given_K = self.__entropies(self.y_test, self.pred)
        homogeneity = 1 - H_K_given_C / H_K if H_K != 0 else 1.0
        completeness = 1 - H_C_given_K / H_C if H_C != 0 else 1.0
        return 2 * (homogeneity * completeness) / (homogeneity + completeness) if (homogeneity + completeness) != 0 else 0.0
    
    
    #predefined metrics
    def homogeneity_score(self):
        return homogeneity_score(self.y_test, self.pred)
    
    def completeness_score(self):
        return completeness_score(self.y_test, self.pred)
    
    def v_measure_score(self):
        return v_measure_score(self.y_test, self.pred)
     
class InternalEvaluations(Evaluations):
    
    def __init__(self, pred, X_test):
        super().__init__(pred)
        self.X_test = X_test
    
    def silhouette(self):
        n = len(self.X_test)
        a, b = np.zeros(n), np.zeros(n)
        
        for i in range(n):
            cluster_label = self.pred[i]
            cluster_points = self.X_test[self.pred == cluster_label]
            
            a[i] = np.mean(np.linalg.norm(cluster_points - self.X_test[i], axis=1))

            min_dist = np.inf
            for j in range(len(np.unique(self.pred))):
                if j != cluster_label:
                    other_cluster_points = self.X_test[self.pred == j]    
                    dist = np.mean(np.linalg.norm(other_cluster_points - self.X_test[i], axis=1))
                    if dist < min_dist:
                        min_dist = dist
            b[i] = min_dist
            
        return np.mean((b - a) / np.maximum(a, b))
        
           
    def davies_bouldin(self):
        n_clusters = len(np.unique(self.pred))
        #computing the controid Ci of each cluster Ci
        cluster_centers = np.array([np.mean(self.X_test[self.pred == i], axis = 0) for i in range(n_clusters)])
        sigma = np.zeros(n_clusters)
        R = np.zeros((n_clusters, n_clusters)) # Creating a 2D array with 'n_clusters' rows and 'n_clusters' columns filled with 0 values

        # computiong the average distance from each point in cluster Ci to its centroid 
        for i in range(n_clusters): 
            cluster_points = self.X_test[self.pred == i]
            centroid_i = cluster_centers[i]
            sigma[i] = np.mean(np.linalg.norm(cluster_points - centroid_i, axis=1))
           
        # ccomputing the similarity Rij between each pair of cluster Ci and Cj 
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    centroid_i = cluster_centers[i]
                    centroid_j = cluster_centers[j]
                    distance_ij = np.linalg.norm(centroid_i - centroid_j)
                    R[i, j] = (sigma[i] + sigma[j]) / distance_ij
                
        return np.mean(np.max(R, axis=1))  
        
    
    def calinski_harabasz(self):
        k = len(np.unique(self.pred))
        scatter_within, scatter_between = [], []
        
        overall_centroid = np.mean(self.X_test, axis=0)
        
        for i in range(k):
            cluster_points = self.X_test[self.pred == i]
            centroid = np.mean(cluster_points, axis=0)
            
            scatter_within.append(np.sum((cluster_points - centroid) ** 2))
            scatter_between.append(len(cluster_points) * np.sum((centroid - overall_centroid) ** 2))

        return (np.sum(scatter_between) / (k - 1)) / (np.sum(scatter_within) / (len(self.X_test) - k)) 
        
    
    def silhouette_score(self):
        return silhouette_score(self.X_test, self.pred)
    
    def davies_bouldin_score(self):
        return davies_bouldin_score(self.X_test, self.pred)
    
    def calinski_harabasz_score(self):
        return calinski_harabasz_score(self.X_test, self.pred)
    