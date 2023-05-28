from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Dataset():
    
    def __init__(self, datadir='data/starcraft_player_data.csv'):
        self.data = pd.read_csv(datadir, na_values='?')
        self.target = self.data.LeagueIndex

        # remove features we're not using for unsupervised learning
        dropped_features = ['Age', 'LeagueIndex', 'GameID']
        self.data.drop(columns=dropped_features, inplace=True)

        # Replace missing values with a placeholder value
        missing_value_placeholder = -1
        self.data = self.data.replace(np.nan, missing_value_placeholder)

        # Winsorize Total hours outliers by replacing outliers with the 95th percentile value
        upp_clip = self.data.TotalHours.quantile(0.95)
        self.data.TotalHours = self.data.TotalHours.clip(upper=upp_clip)
        
        # Scale data
        self.standard_features = ['UniqueUnitsMade', 'UniqueHotkeys', 'TotalHours']
        self.minmax_features = [feat for feat in self.data.columns if feat not in self.standard_features]
        self.scaled_data, _, _ = self._scale_features()
        
        # Placeholder for scalers
        self.min_max_scaler = self.standard_scaler = None
        
        # Impute missing values of HoursPerWeek and TotalHours
        self._fillin_values()
        
        # Perform Feature engineering and removes features that will not be used for training/testing
        self._featurize()
        
        
    def _scale_features(self):
        '''
            Performs custom made scaling to specific features in the dataset. This
            function allows for flexibility in case there are different type of
            distributions on the dataset.
        '''
        # Perform Min-Max scaling on selected features
        min_max_scaler = MinMaxScaler()
        data_scaled_minmax = min_max_scaler.fit_transform(self.data[self.minmax_features])

        # Perform Standardization on selected features
        standard_scaler = StandardScaler()
        data_scaled_standard = standard_scaler.fit_transform(self.data[self.standard_features])

        # Combine the scaled features with the remaining unchanged features
        scaled_data = self.data.copy()
        scaled_data[self.minmax_features] = data_scaled_minmax
        scaled_data[self.standard_features] = data_scaled_standard

        return scaled_data, min_max_scaler, standard_scaler
    
    def _get_optimal_k(self):
        k_values = range(2, 11)
        inertias = []

        # Perform K-means clustering for each value of k
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=43)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
            
        return k_values[np.argmin(np.diff(inertias)) + 1]
    
    def _fillin_values(self):
        num_clusters = self._get_optimal_k()
        kmeans = KMeans(n_clusters=num_clusters, random_state=53)
        kmeans.fit(self.scaled_data)

        self.data['Cluster'] = kmeans.labels_
        for feature in self.data.columns:
            for cluster_id in range(num_clusters):
                cluster_values = self.data[self.data['Cluster'] == cluster_id][feature]
                cluster_median = np.median(cluster_values)

                self.data.loc[(self.data[feature] == -1) & (self.data['Cluster'] == cluster_id), feature] = cluster_median
                
    def _featurize(self):
        # Ratio features
        self.data['PlaytimeIntensity'] = self.data.HoursPerWeek / self.data.TotalHours
        self.data['SelectByHotkeysRatio'] = self.data.SelectByHotkeys / self.data.APM
        self.data['PACsPerActionRatio'] = self.data.NumberOfPACs / self.data.APM
        self.data['WorkersToUnitsRatio'] = self.data.WorkersMade / (self.data.WorkersMade + self.data.UniqueUnitsMade + self.data.ComplexUnitsMade)
        self.data['EconomyManagement'] = self.data.WorkersMade / self.data.TotalMapExplored

        # Relationship features
        self.data['APM_TotalMapExplored'] = self.data.APM * self.data.TotalMapExplored
        
        removed_features = ['UniqueUnitsMade', 'ComplexAbilitiesUsed', 'TotalMapExplored', 'HoursPerWeek']
        self.data.drop(columns=removed_features, inplace=True)
        self.standard_features = ['UniqueHotkeys', 'TotalHours']
        self.minmax_features = [feat for feat in self.data.columns if feat not in self.standard_features]
        self.scaled_data, self.min_max_scaler, self.standard_scaler = self._scale_features()
    
    def get_data(self, scaled=False):
        '''
            Returns a copy of the entire dataset.
            
            Params:
                scaled - whether to return the scaled dataset or the non scaled version.
        '''
        data = self.scaled_data.copy() if scaled else self.data.copy()
        data.drop('Cluster', axis=1, inplace=True)
        data['LeagueIndex'] = self.target.copy()
        return data
        
    def get_train_test_data(self, test_size=0.2, seed=53, scaled=False):
        '''
            Performs train test split on the dataset and return the training and test batches randomly shuffled.
            
            Params:
                test_size - size of the test batch as a fraction.
                seed      - random seed value.
                scaled    - whether to return the scaled or nonscaled version of the dataset.
        '''
        data = self.get_data(scaled)
        target = data.LeagueIndex
        X_train, X_test, y_train, y_test = train_test_split(data.drop('LeagueIndex', axis=1), target, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test