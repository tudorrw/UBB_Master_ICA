from dataset import Dataset
import numpy as np
from knn import KNearestNeighbors
from kfold_cross_validation import KFoldCrossValidation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

def find_optimal_k(X_train, y_train, X_test, y_test, max_k=15):
    best_k = 1
    best_acc = 0
    k_results = []

    print("\n--- Hyperparameter Optimization: Finding Optimal K ---")
    for k in range(1, max_k + 1, 2):
        model = KNearestNeighbors(k=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = np.sum(preds == y_test) / len(y_test)
        k_results.append((k, acc))
        print(f"Testing k={k} | Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_k = k
            
    print(f"Optimal K identified: {best_k} with {best_acc*100:.2f}% accuracy")
    return best_k

if __name__ == "__main__":
    PATH = "./dataset/"
    TRAIN_FILENAME = f'{PATH}train.csv'
    TEST_FILENAME = f'{PATH}test.csv'

    try:
        dataset_loader = Dataset(TRAIN_FILENAME, TEST_FILENAME)
        X, y = dataset_loader.preprocess(scale=True)
        print("Preprocessing completed successfully.")

        subset_size = 10000
        X_sub, y_sub = X[:subset_size], y[:subset_size]
        
        split_idx = int(subset_size * 0.8)
        X_train, X_test = X_sub[:split_idx], X_sub[split_idx:]
        y_train, y_test = y_sub[:split_idx], y_sub[split_idx:]

        X_tune = X_train[:2000]
        y_tune = y_train[:2000]
        X_val = X_test[:500]
        y_val = y_test[:500]

        optimal_k = find_optimal_k(X_tune, y_tune, X_val, y_val, max_k=11)

        print(f"Comparison started on {subset_size} samples.")

        print("\nRunning Scikit-Learn KNN (Library)...")
        sk_start = time.time()
        sk_model = KNeighborsClassifier(n_neighbors=optimal_k)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_end = time.time()

        sk_acc = accuracy_score(y_test, sk_preds)
        sk_f1 = f1_score(y_test, sk_preds)

        print("Running Manual KNN (From Scratch)...")
        my_start = time.time()
        my_model = KNearestNeighbors(k=optimal_k)
        my_model.fit(X_train, y_train)
        my_preds = my_model.predict(X_test)
        my_end = time.time()

        my_acc = np.sum(my_preds == y_test) / len(y_test)

        my_probs = my_model.predict_proba(X_test)
        
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        sk_probs = sk_model.predict_proba(X_test)[:, 1]
        
        sk_auc = roc_auc_score(y_test, sk_probs)
        my_auc = roc_auc_score(y_test, my_probs)

        print("\n" + "="*70)
        print(f"{'Metric':<20} | {'Sklearn KNN':<15} | {'Manual KNN':<15}")
        print("-" * 70)
        print(f"{'Accuracy':<20} | {sk_acc:<15.4f} | {my_acc:<15.4f}")
        print(f"{'AUC-ROC':<20} | {sk_auc:<15.4f} | {my_auc:<15.4f}")
        print(f"{'Execution Time':<20} | {sk_end - sk_start:<15.2f}s | {my_end - my_start:<15.2f}s")
        print("="*70)

        plot_confusion_matrix(y_test, my_preds, "Manual KNN (Scratch)")


        cv_manager = KFoldCrossValidation(n_folds=5)
        my_knn_model = KNearestNeighbors(k=optimal_k)
        cv_manager.evaluate(my_knn_model, X_sub, y_sub)

    except FileNotFoundError:
        print(f"Files not found in {PATH}. Please check the folder path!")
    except Exception as e:
        print(f"An error occurred: {e}")