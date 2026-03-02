import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class Dataset:
    def __init__(self, train_filename, test_filename):
        train_set = pd.read_csv(train_filename)
        test_set = pd.read_csv(test_filename)
        self.data = pd.concat([train_set, test_set])

    def __remove_undefined_values(self):
        for col in self.data.columns[8:21]:
            value_counts = dict(self.data[col].value_counts())
            if 0 in value_counts:
                self.data = self.data[self.data[col] != 0]

    def __drop_eco_plus_cat(self):
        self.data = self.data[self.data['Class'] != 'Eco Plus']

    def __drop_columns(self):
        self.data = self.data.drop(self.data.columns[[0, 1]], axis=1)
        self.data = self.data.drop(['Arrival Delay in Minutes'], axis=1)

    def __encode_features_of_type_object(self):

        object_cols = self.data.select_dtypes(include='object').columns
        le = LabelEncoder()
        for col in object_cols:
            print(self.data[col].value_counts())
            self.data[col] = le.fit_transform(self.data[col])
            # print(self.data[col].value_counts())
            
            
    def __scale_input_features(self, X):
        return MinMaxScaler().fit_transform(X)
        # return StandardScaler().fit_transform(X)


    def preprocesses(self, scale=True):
        self.__remove_undefined_values()
        # self.drop_eco_plus_cat()
        self.__drop_columns()
        self.__encode_features_of_type_object()
        
        X = self.data.drop('satisfaction', axis=1)
        if scale==True:
            X = self.__scale_input_features(X)
        
        y = self.data['satisfaction']

        return X, y

