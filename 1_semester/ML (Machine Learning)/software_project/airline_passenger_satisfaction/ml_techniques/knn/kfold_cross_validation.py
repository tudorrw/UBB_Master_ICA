import numpy as np
import pandas as pd

class KFoldCrossValidation: 
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def __calculate_auc(self, y_true, y_probs):
        pos = y_probs[y_true == 1]
        neg = y_probs[y_true == 0]
        if len(pos) == 0 or len(neg) == 0: return 0
        count = 0
        for p in pos:
            for n in neg:
                if p > n: count += 1
                elif p == n: count += 0.5
        return count / (len(pos) * len(neg))

    def __calculate_auprc(self, y_true, y_probs):
        desc_idx = np.argsort(y_probs)[::-1]
        y_true_sorted = y_true[desc_idx]
        precisions, recalls = [], []
        tp, fp = 0, 0
        total_pos = np.sum(y_true)
        if total_pos == 0: return 0
        for val in y_true_sorted:
            if val == 1: tp += 1
            else: fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / total_pos)
        return np.trapz(precisions, recalls)

    def __calculate_metrics(self, y_true, y_pred, y_probs):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity = recall 
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        auc_score = self.__calculate_auc(y_true, y_probs)
        auprc_score = self.__calculate_auprc(y_true, y_probs)
        
        return accuracy, precision, recall, sensitivity, f_measure, auc_score, auprc_score

    def evaluate(self, model, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        indices = np.arange(len(X))
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)
        
        print(f"Starting {self.n_folds}-Fold CV for {model.__class__.__name__}")
        print("-" * 100)

        results = []
        for i in range(self.n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            metrics = self.__calculate_metrics(y_test, predictions, probabilities)
            results.append(metrics)
            
            print(f"Fold {i+1} | Acc: {metrics[0]:.4f} | Prec: {metrics[1]:.4f} | Rec: {metrics[2]:.4f} | AUC: {metrics[5]:.4f}")

        results = np.array(results)
        print("-" * 70)
        print(f"FINAL STATISTICAL ANALYSIS:")
        
        metric_names = ["Accuracy", "Precision", "Recall", "Sensitivity", "F-measure", "AUC-ROC", "AUPRC"]
        
        for idx, name in enumerate(metric_names):
            mean_val = np.mean(results[:, idx])
            std_val = np.std(results[:, idx])
            print(f"Mean {name:<12}: {mean_val:.4f} (+/- {std_val:.4f})")