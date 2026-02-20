import numpy as np

class KMeansClustering:
    def __init__(self, k, max_iter=200, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids): # one data point and all centroids
        return np.sqrt(np.sum((centroids - data_point)**2, axis = 1)) # the distance between every single data point and all the centroids
    
    
    def fit(self, X):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        #boundaries for the uniform distribution
        #uniform - you give it a min and a max and every value between min and max has the same chance of being chosen
        #for the centroids we generate for every dimension 
        # we keep coordinates of the centroids within the min and max of the respective dimension

        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), # for each value, the minimum value in X and the maximum for every dimension
                                           size = (self.k, X.shape[1])) 
        y = None 
        for _ in range(self.max_iter): # iterate max_iter times
            y = [] # cluster labels: y labels for the X data points
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids) #a np array of distances
                cluster_num = np.argmin(distances) #index of the smallest value: the closest centroid, the cluster that has the smallest distance to this point
                y.append(cluster_num)
            
            y = np.array(y)

            #repositioning the centroids based on the labels

            cluster_indices = [] #list of lists: for each cluster which of the data points belong to that cluster
            
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) #for every point in X which cluster it belongs to
                
            cluster_centers = [] 
            #repositioning the centroids based on the labels
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:  # for empty clusters
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis = 0)[0]) # average position; the indices of the elements that belong to that cluster
                    # , take all their positions, compute the mean positions and take the mean, so the first value [0]
            
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001: # if the difference between the old centroids and the new centroids is smaller than a threshold, it will break
                break
            else:
                self.centroids = np.array(cluster_centers) #repositioning cluster centers
        
        self.labels_ = y
        self.inertia_ = self._calculate_inertia(X, y)
        return y 

    def _calculate_inertia(self, X, labels):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                squared_distances = np.sum((cluster_points - self.centroids[i])**2, axis=1)
                inertia += np.sum(squared_distances)
        return inertia
        
        
    def predict(self, X):
        predictions = []
        for data_point in X:
            distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
            predictions.append(np.argmin(distances))
        return np.array(predictions)
    
    @property
    def cluster_centers_(self):
        return self.centroids    
    
        
    