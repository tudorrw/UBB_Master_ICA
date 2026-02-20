from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP

class DimensionalityReduction:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca_model = None
        self.umap_model = None
    
    def pca(self, X, y, n_components=3):
        # Fit PCA
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_model.fit(X)
        X_transformed = self.pca_model.transform(X)
        column_names = [f'col{i+1}' for i in range(n_components)]
        
        plt.figure(figsize=(7,6))
        sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y, s=25, alpha=0.75)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Projection (2D)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(X_transformed, columns=column_names)
    
    
    def umap(self, X, y, n_components=3):
        self.umap_model = UMAP(n_neighbors=100, n_components=2, metric='euclidean', n_epochs=100, learning_rate=0.1, init='spectral',
                 min_dist=0.1, spread=1.0, low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1,
                 repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, random_state=self.random_state,  
                 angular_rp_forest=False, target_n_neighbors=-1, transform_seed=3, verbose=True, unique=False)
        
        X_transformed = self.umap_model.fit_transform(X)
        
        # Create column names if not provided
        column_names = [f'col{i+1}' for i in range(n_components)]
            
        plt.figure(figsize=(14,8))
        sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=y)
        plt.title('UMAP Components Plot')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.show()
        
        return pd.DataFrame(X_transformed, columns=column_names)
    
  