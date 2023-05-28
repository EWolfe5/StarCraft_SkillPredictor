from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Dataset():
    
    LIMIT = 10000
    FEATURES = [
        'LeagueIndex', 'APM', 'ActionLatency', 'AssignToHotkeys', 'GapBetweenPACs',
        'NumberOfPACs', 'SelectByHotkeys', 'TotalHours', 'UniqueHotkeys'
    ]
    
    def __init__(self, datadir='data/starcraft_player_data.csv', scaler=MinMaxScaler):
        self.data = pd.read_csv(datadir, na_values='?')
        # self.data = self.data.loc[:, self.FEATURES]
        self.data.drop(columns=['GameID', 'Age', 'HoursPerWeek'], inplace=True)
        # Consolidate classes
        self.data.LeagueIndex.replace({7:6, 8:6, 6:6}, inplace=True)
        # Fill in missing values and handle outliers
        self._fillin_values()
        # Store target variable
        self.target = self.data.LeagueIndex.copy()
        self.data.drop('LeagueIndex', axis=1, inplace=True)
        self.scaler = scaler()
        self.scaled_data = self._scale_features()
        
        
    def _scale_features(self):
        '''
            Performs custom made scaling to specific features in the dataset. This
            function allows for flexibility in case there are different type of
            distributions on the dataset.
        '''
        scaled_df = pd.DataFrame(self.scaler.fit_transform(np.log1p(self.data)), 
                                 columns=self.data.columns)
        return scaled_df
    
    
    def _fillin_values(self):
        # Handle outliers in total hours
        replaceindx = self.data.query('TotalHours > %i' % self.LIMIT).index
        replace_values = self.data[['LeagueIndex', 'TotalHours']].groupby('LeagueIndex').agg(lambda x: np.nanpercentile(x, 99)).TotalHours
        replace_values = np.array(replace_values[np.array(self.data.loc[replaceindx].LeagueIndex)])
        self.data.loc[replaceindx, 'TotalHours'] = replace_values

        # Replace Total Hours with classes 99th percentile
        th_replace_value = np.percentile(
            self.data.loc[(self.data.LeagueIndex == 6) & (~self.data.TotalHours.isnull()), 'TotalHours'], 
            99
        )
        self.data.TotalHours.replace(np.nan, th_replace_value, inplace=True)

    
    def get_data(self, scaled=False):
        '''
            Returns a copy of the entire dataset.
            
            Params:
                scaled - whether to return the scaled dataset or the non scaled version.
        '''
        data = self.scaled_data.copy() if scaled else self.data.copy()
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