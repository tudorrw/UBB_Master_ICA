import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, train_filename, test_filename):
        train_set = pd.read_csv(train_filename)
        test_set = pd.read_csv(test_filename)
        self.data = pd.concat([train_set, test_set]).reset_index(drop=True)

    def __remove_undefined_values(self):
        for col in self.data.columns[8:21]:
            self.data = self.data[self.data[col] != 0]

    def __drop_columns(self):
        self.data = self.data.drop(self.data.columns[[0, 1]], axis=1)
        if 'Arrival Delay in Minutes' in self.data.columns:
            self.data = self.data.drop(['Arrival Delay in Minutes'], axis=1)

    def __manual_label_encoding(self):
        object_cols = self.data.select_dtypes(include='object').columns
        for col in object_cols:
            unique_values = self.data[col].unique()
            mapping = {val: i for i, val in enumerate(sorted(unique_values))}
            self.data[col] = self.data[col].map(mapping)
            #print(f"Mapping for {col}: {mapping}")

    def __manual_min_max_scaling(self, X):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_scaled = (X - X_min) / (X_max - X_min + 1e-10)
        return X_scaled

    def preprocess(self, scale=True):
        self.__remove_undefined_values()
        self.__drop_columns()
        self.__manual_label_encoding()
        
        
        X = self.data.drop('satisfaction', axis=1)
        y = self.data['satisfaction']

        if scale:
            X = self.__manual_min_max_scaling(X)
        
        return X.to_numpy(), y.to_numpy()