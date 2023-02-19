import pandas as pd
import json
import numpy as np

class PreProcessing:
    """ Deal with missing values, drop useless columns and apply feature engineering"""
    
    def __init__(self, data: pd.DataFrame, features_to_drop:list[str], features_to_fill_by_median:list[str],\
         features_to_remove_nan_values:list[str], features_to_log:list[str]) -> None:
        self.data = data.copy()
        self.features_to_drop = features_to_drop
        self.features_to_fill_by_median = features_to_fill_by_median
        self.features_to_remove_nan_values = features_to_remove_nan_values
        self.features_to_log = features_to_log

    def preprocess(self, is_for_train: bool=False) -> pd.DataFrame:
        """ Drop dupclicated, Apply missing values imputation

        Args:
            is_for_train (bool): drop rows with missing values for important features.
            Defaults to False.

        Returns:
            pd.DataFrame: preprocessed data
        """

        print(f'''\n{'-'*20} Preprocessing data  {'-'*20}''')
        # drop duplicates
        print(f'''\nDrop duplicates''')
        self.data.drop_duplicates(inplace=True)
        # drop useless cols
        print(f'Drop redundant features, features with more than 80% missing values and features which cannot be known in advance of the match: {self.features_to_drop}')
        self.data.drop(columns=self.features_to_drop, axis=1, inplace=True)
        # fill nan values
        self._fill_nan_values(is_for_train)
        print('\nSplit Date into Year, Month and Day and extract new feature "is_weekend"')
        self._split_date()
        print(f'\nLog numerical features that are right skewed : {self.features_to_log}')
        self._log_features()
        print(f'\nData shape after cleaning and preprocessing : {self.data.shape}')
        print(f"\nRemained features : {list(self.data.columns)}")
        return self.data
        

    def _display_columns_with_nan_values(self) -> None:
        # sourcery skip: dict-comprehension
        """
        Find columns that contain nan values.
        """
        nb_rows = len(self.data)
        features_with_nan_values: dict[str, float] = {}
        for c in self.data.columns :
            if self.data[c].count() < nb_rows :
                features_with_nan_values[c] = round((1 - self.data[c].count() / nb_rows) * 100, 2)
           
        features_with_nan_values = {
            k: f'{str(v)}%'
            for k, v in sorted(
                features_with_nan_values.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        print(f'\nFeatures with missing values : {json.dumps(features_with_nan_values, indent = 4)}')   
        return None
 
    def _fill_nan_values(self, is_for_train: bool=False) -> None:
        """ Impute missing values

        Args:
            is_for_train (bool): drop rows with missing values for important features
            Defaults to False.
        """
        self.data.replace(to_replace=['None'], value=None, inplace=True)
        self._display_columns_with_nan_values()

        print(f'\nImpute missing values with median for these features : {self.features_to_fill_by_median}')
        print(f'\nRemove rows with missing values for important features: {self.features_to_remove_nan_values}')
              
        # fill nan values by a median
        for c in self.features_to_fill_by_median:
            if self.data[c].count() > 1 :
                self.data[c] = self.data[c].fillna((self.data[c].median()))
            else: 
                self.data[c] = self.data[c].fillna(0)
        ## fillna for pi_hand with 'U' as there is already an unkonwn category
        self.data['p1_hand'] = self.data['p1_hand'].fillna('U')
        self.data['p2_hand'] = self.data['p2_hand'].fillna('U')
        ## fillna for surface with 'A' as there is already an unkonwn category
        self.data['tourney_level'] = self.data['tourney_level'].fillna('A')
        self.data['tourney_level'] = self.data['tourney_level'].fillna('A')
        
        # remove empty rows for important features in train set
        if is_for_train:
            self.data.dropna(how='any',inplace=True) 

        self.data.reset_index(inplace=True, drop=True)
        self._display_columns_with_nan_values()
        return None
    
    def _split_date(self) -> None:
        """  Extract the year, the month and the day from the column tourney date. """
        self.data['tourney_date'] = pd.to_datetime(self.data['tourney_date'], format="%Y%m%d")
        self.data["day"] = self.data["tourney_date"].map(lambda x: x.day)
        self.data["month"] = self.data["tourney_date"].map(lambda x: x.month)
        self.data["year"] = self.data["tourney_date"].map(lambda x: x.year)
        self.data['is_weekend'] = self.data["tourney_date"].map(lambda x : 1 if x.weekday() >= 5 else 0)
        return

    def _log_features(self) -> None:
        """Log numerical features that are right skewed"""
        for feature in self.features_to_log:
            self.data[feature] = np.log(self.data[feature] + np.finfo(float).eps)



