from sklearn.cluster import KMeans
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dataset import Dataset
from kmeans import KMeansClustering
from kfold_cross_validation import KFoldCrossValidation
from dimensionality_reduction import DimensionalityReduction

def elbow_method(X):
    # X = X.values
    cluster_numbers = range(2, 15)
    inertia = []
    for k in tqdm(cluster_numbers):
        # kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data_pca)
        kmeans1 = KMeansClustering(k=k, max_iter=300)
        kmeans1.fit(X)
        inertia.append(kmeans1.inertia_)

    print("Plotting inertia vs number of clusters...", inertia)
    plt.plot(cluster_numbers, inertia, marker='o')
    plt.show()
def main():
    PATH = "../../dataset/"
    TRAIN_FILENAME = f'{PATH}train.csv'
    TEST_FILENAME = f'{PATH}test.csv'
    
    dataset = Dataset(TRAIN_FILENAME, TEST_FILENAME)
    X_scaled, y = dataset.preprocesses(scale=True)
    
    dataset1 = Dataset(TRAIN_FILENAME, TEST_FILENAME)
    X, _ = dataset1.preprocesses(scale=False)

    # # data_pca = DimensionalityReduction(random_state=42).pca(X_scaled, y, n_components=2)
    # data_umap = DimensionalityReduction(random_state=3).umap(X_scaled, y, n_components=2)
    
    elbow_method(X_scaled)

    K = 8
    custom_kmeans = KMeansClustering(k=K, max_iter=300)
    sklearn_kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    
    evaluator = KFoldCrossValidation(n_folds=5, random_state=42, original_data=X)

    print("Evaluating Custom Implementation.")
    evaluator.evaluate(custom_kmeans, X_scaled, y)
    
    print("Evaluating Scikit-Learn Implementation.")
    evaluator.evaluate(sklearn_kmeans, X_scaled, y)

if __name__ == '__main__':
    main()