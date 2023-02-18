import pandas as pd
import json


def load_data(data_path: str) -> pd.DataFrame:
    """ Load data with the path provided in config.yaml
    Args:
        data_path (str): path of data csv

    Returns:
        pd.DataFrame: loaded data
    """
    # load data :
    print(f''' {'-'*20} Loading data  {'-'*20}''')
    try:
        data = pd.read_csv(data_path, sep = ';')
        print('Data shape : ', data.shape)
        return data
    except Exception as e:
        raise "No data found with the path provided in config.yaml" from e

class PreProcessing:
    """ Deal with nan values, drop useless columns"""
    
    def __init__(self, data: pd.DataFrame, features_to_drop:list[str], features_to_fill_by_median:list[str],\
         features_to_remove_nan_values:list[str]) -> None:
        self.data = data.copy()
        self.features_to_drop = features_to_drop
        self.features_to_fill_by_median = features_to_fill_by_median
        self.features_to_remove_nan_values = features_to_remove_nan_values

    def preprocess(self) -> pd.DataFrame:
        """ Drop dupclicated, Apply missing values imputation """

        print(f'''\n{'-'*20} Preprocessing data  {'-'*20}''')
        # drop duplicates
        print(f'''\nDrop duplicates''')
        self.data.drop_duplicates(inplace=True)
        # drop useless cols
        print(f'Drop redundant features, features with more than 80% missing values and features which cannot be known in advance of the match: {self.features_to_drop}')
        self.data.drop(columns=self.features_to_drop, axis=1, inplace=True)
        # fill nan values
        self._fill_nan_values()
        print('\nSplit Date into Year, Month and Day and extract new feature "is_weekend"')
        self._split_date()
        print(f'\nData shape after cleaning and preprocessing : {self.data.shape}')
        print(f"\nremained features : {list(self.data.columns)}")
        return self.data
        

    def _display_columns_with_nan_values(self) -> None:
        """
        Find columns that contain nan values.
        """
        nb_rows = len(self.data)
        features_with_nan_values: dict[str, float] = {
            c: round((1 - self.data[c].count() / nb_rows) * 100, 2)
            for c in self.data.columns
            if self.data[c].count() < nb_rows
        }
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
 
    def _fill_nan_values(self) -> None:
        """  Fill nan values """
        
        self._display_columns_with_nan_values()

        print(f'\nImpute missing values with median for these features : {self.features_to_fill_by_median}')
        print(f'\nRemove rows with missing values for important features: {self.features_to_remove_nan_values}')
              
        # fill nan values by a median
        for c in self.features_to_fill_by_median:
            self.data[c] = self.data[c].fillna((self.data[c].median()))
        
        ## fillna for pi_hand with 'U' as there is already an unkonwn category
        self.data['p1_hand'] = self.data['p1_hand'].fillna('U')
        self.data['p2_hand'] = self.data['p2_hand'].fillna('U')
        self.data.reset_index(inplace=True, drop=True)

        # remove empty rows for important features
        self.data.replace(to_replace=['None'], value=None, inplace=True)
        self.data.dropna(how='any',inplace=True) 

        self._display_columns_with_nan_values()
        return None
    
    def _split_date(self) -> None:
        """  Extract the year, the month and the day from the column tourney date.
        """
        self.data['tourney_date'] = pd.to_datetime(self.data['tourney_date'], format="%Y%m%d")
        self.data["day"] = self.data["tourney_date"].map(lambda x: x.day)
        self.data["month"] = self.data["tourney_date"].map(lambda x: x.month)
        self.data["year"] = self.data["tourney_date"].map(lambda x: x.year)
        self.data['is_weekend'] = self.data["tourney_date"].map(lambda x : 1 if x.weekday() >= 5 else 0 )
        return