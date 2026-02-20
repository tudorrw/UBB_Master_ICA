
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from scipy import stats

from evaluations import ExternalEvaluations, InternalEvaluations

class KFoldCrossValidation: 
    
    def __init__(self, n_folds=5, random_state=42, original_data=None):
        self.n_folds = n_folds
        self.random_state = random_state
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)   
        self.fold_idx = 0
        # Store original_data as DataFrame for heatmap display
        if original_data is not None:
            if isinstance(original_data, pd.DataFrame):
                self.original_data = original_data.copy()
            elif isinstance(original_data, np.ndarray):
                self.original_data = pd.DataFrame(original_data)
            else:
                self.original_data = pd.DataFrame(original_data)
        else:
            self.original_data = None
        
        # Store metrics across folds for confidence interval calculation
        self.metrics_history = {
            'v_measure': [],
            'homogeneity': [],
            'completeness': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }  
           
    def __external_evaluations(self, pred, y_test):
        
        ext_eval = ExternalEvaluations(pred, y_test)

        # from build-in libraries
        # v_score1 = ext_eval.v_measure_score()
        # homogeneity1 = ext_eval.homogeneity_score()
        # completeness1 = ext_eval.completeness_score()
        # print(f"Fold {self.fold_idx}: V-Measure = {v_score1:.4f} Homogeneity = {homogeneity1:.4f} | Completeness = {completeness1:.4f}")
        
        # mathematical calculation        
        v_score = ext_eval.v_measure()
        homogeneity = ext_eval.homogeneity()
        completeness = ext_eval.completeness()
        print(f"Fold {self.fold_idx}: V-Measure = {v_score:.4f} Homogeneity = {homogeneity:.4f} | Completeness = {completeness:.4f}")
        
        # Store metrics for confidence interval calculation
        self.metrics_history['v_measure'].append(v_score)
        self.metrics_history['homogeneity'].append(homogeneity)
        self.metrics_history['completeness'].append(completeness)
        
        return v_score, homogeneity, completeness
      
       
    def __internal_evaluations(self, pred, X_test):
        
        int_eval = InternalEvaluations(pred, X_test)  
            # from build-in libraries
        # sil_score = int_eval.silhouette_score()
        # db_score = int_eval.davies_bouldin_score()
        # ch_score = int_eval.calinski_harabasz_score()
        # print(f"Fold {self.fold_idx}: Silhoutte = {sil_score:.4f} Davies Bouldin = {db_score:.4f} | Calinski Harabasz = {ch_score:.4f}")

        sil = int_eval.silhouette()
        db = int_eval.davies_bouldin()
        ch = int_eval.calinski_harabasz()

        print(f"Fold {self.fold_idx}: Silhoutte = {sil:.4f} Davies Bouldin = {db:.4f} | Calinski Harabasz = {ch:.4f}")
        
        # Store metrics for confidence interval calculation
        self.metrics_history['silhouette'].append(sil)
        self.metrics_history['davies_bouldin'].append(db)
        self.metrics_history['calinski_harabasz'].append(ch)
        
        return sil, db, ch

    def __plot_fold(self, X, y, centers, preds):
        plt.figure(figsize=(7,6))
        sns.scatterplot(x=X[:,0], y=X[:,1], hue=preds, s=40, alpha=0.6, palette="viridis", edgecolor='w')
        plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), cmap="viridis", marker="*", s=350, edgecolor='black', label="Centroids", linewidth=1.5)
        plt.xlabel('1')
        plt.ylabel('2')
        plt.title('Projection (2D)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def __display_heatmap(self, pred, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()
        
        X['Cluster'] = pred
        cluster = X.groupby(['Cluster']).mean().T
        cluster_norm = cluster.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else x, axis=1)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cluster_norm, annot=cluster, cmap='RdYlGn', fmt=".4f", linewidths=.5)
        plt.title(f'Fold {self.fold_idx} - Cluster Means')
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    

    def __calculate_confidence_intervals(self, confidence=0.95):
        
        # Calculate confidence intervals using Central Limit Theorem.
        # Formula: CI = μ ± α, where α = z * (σ / √k)
        if confidence == 0.95:
            z_score = 1.96
        elif confidence == 0.99:
            z_score = 2.576
        elif confidence == 0.90:
            z_score = 1.645
        else:
            # Use scipy for other confidence levels
            z_score = stats.norm.ppf((1 + confidence) / 2)
        
        results = {}
        k = self.n_folds
        
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1)  # Sample standard deviation
                margin_of_error = z_score * (std / np.sqrt(k))
                
                results[metric_name] = {
                    'mean': mean,
                    'std': std,
                    'ci_lower': mean - margin_of_error,
                    'ci_upper': mean + margin_of_error,
                    'margin_of_error': margin_of_error
                }
        
        return results
    
    def __print_confidence_intervals(self, confidence=0.95):
        """
        Print confidence intervals for all metrics in a formatted way.
        """
        ci_results = self.__calculate_confidence_intervals(confidence)
        
        print("\n" + "="*80)
        print(f"CONFIDENCE INTERVALS ({int(confidence*100)}% CI) - {self.n_folds}-Fold Cross-Validation")
        print("="*80)
        print(f"Based on Central Limit Theorem: CI = μ ± {1.96 if confidence==0.95 else 'z'} * (σ / √{self.n_folds})")
        print("-"*80)
        
        # Group by external and internal metrics
        external_metrics = ['v_measure', 'homogeneity', 'completeness']
        internal_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        
        print("\nEXTERNAL EVALUATION METRICS:")
        print("-"*80)
        for metric in external_metrics:
            if metric in ci_results:
                res = ci_results[metric]
                print(f"{metric.upper():20s}: std: {res['std']:.4f}, mean: {res['mean']:.4f} ± margin of error: {res['margin_of_error']:.4f} "
                      f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
        
        print("\nINTERNAL EVALUATION METRICS:")
        print("-"*80)
        for metric in internal_metrics:
            if metric in ci_results:
                res = ci_results[metric]
                print(f"{metric.upper():20s}: std: {res['std']:.4f}, mean: {res['mean']:.4f} ± margin of error: {res['margin_of_error']:.4f} "
                      f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
        
        print("="*80 + "\n")
        
        return ci_results
    
    def __plot_metrics_across_folds(self):
        """
        Visualize how each metric varies across folds.
        """
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        metric_labels = {
            'v_measure': 'V-Measure',
            'homogeneity': 'Homogeneity',
            'completeness': 'Completeness',
            'silhouette': 'Silhouette Score',
            'davies_bouldin': 'Davies-Bouldin Index',
            'calinski_harabasz': 'Calinski-Harabasz Score'
        }
        
        for idx, (metric_name, values) in enumerate(self.metrics_history.items()):
            if len(values) > 0 and idx < len(axes):
                ax = axes[idx]
                folds = np.arange(1, len(values) + 1)
                mean_val = np.mean(values)
                
                ax.plot(folds, values, marker='o', linewidth=2, markersize=8, 
                       color='steelblue', label='Fold Values')
                ax.axhline(y=mean_val, color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {mean_val:.4f}')
                ax.fill_between(folds, values, mean_val, alpha=0.3, color='lightblue')
                
                ax.set_xlabel('Fold', fontsize=10, fontweight='bold')
                ax.set_ylabel('Score', fontsize=10, fontweight='bold')
                ax.set_title(metric_labels.get(metric_name, metric_name), 
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(fontsize=8)
                ax.set_xticks(folds)
        
        # Hide unused subplots
        for idx in range(len(self.metrics_history), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Metric Variation Across {self.n_folds} Folds', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, model, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        print(f"Starting {self.n_folds}-Fold CV for {model.__class__.__name__}-")
        
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(X), 1):
            self.fold_idx = fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_test = y[test_idx]
            
            model.fit(X_train)
            predictions = model.predict(X_test)
            
            centers = getattr(model, 'cluster_centers_', None)
            # Use original data for heatmap if available, otherwise use scaled data
            
            X_original_test = self.original_data.iloc[test_idx].copy()
            self.__display_heatmap(predictions, X_original_test)

            self.__external_evaluations(predictions, y_test)
            self.__internal_evaluations(predictions, X_test)
            

            # if centers is not None:
            #     self.__plot_fold(X_test, y_test, centers, predictions)
        
        # After all folds, calculate and display confidence intervals
        print("\n" + "="*80)
        print("Cross-Validation Complete!")
        print("="*80)
        

        # Plot metrics across folds
        self.__print_confidence_intervals(confidence=0.95)
        self.__plot_metrics_across_folds()

