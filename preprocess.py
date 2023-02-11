import pandas as pd
import json

class PreProcessing:
    """Deal with nan values, drop useless columns"""
    
    def __init__(self, data: pd.DataFrame, features_to_drop:list[str], features_to_fill_by_median:list[str],\
         features_to_fill_by_new_category:list[str]) -> None:
        self.data = data.copy()
        self.features_to_drop = features_to_drop
        self.features_to_fill_by_median = features_to_fill_by_median
        self.features_to_fill_by_new_category = features_to_fill_by_new_category

    def get_columns_with_nan_values(self) -> dict[str, str]:
        """
        Find columns that contain nan values.
        """
        nb_lines = self.data.shape[0]
        features_with_nan_values: dict[str, float] = {
            c: round((1 - self.data[c].count() / self.data.shape[0]) * 100, 2)
            for c in self.data.columns
            if self.data[c].count() < nb_lines
        }
        return {
            k: f'{str(v)}%'
            for k, v in sorted(
                features_with_nan_values.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
                
 
    def _fill_nan_values(self) -> None:
        """
        Fill nan values
        """
        features_with_nan_values = self.get_columns_with_nan_values()
        print(f'\nFeatures with missing values : {json.dumps(features_with_nan_values, indent = 4)}')
        print(f'\nFeatures to fill missing values with median : {self.features_to_fill_by_median}')
        print(f'\nFeatures to fill missing values with a new category unknown : {self.features_to_fill_by_new_category}')
              
        for c in self.features_to_fill_by_median:
            self.data[c] = self.data[c].fillna((self.data[c].median()))
                        
        for c in self.features_to_fill_by_new_category:
            # fill nan values by a new category unknown(unk)
            self.data[c] = self.data[c].astype('str').replace("nan", "unk") 
        
        ## drop the value of the hand column
        self.data['p1_hand'] = self.data['p1_hand'].fillna('U')
        self.data['p2_hand'] = self.data['p2_hand'].fillna('U')
        self.data.reset_index(inplace=True, drop=True)
        
        features_with_nan_values = self.get_columns_with_nan_values()
        print(f'\nFeatures with missing values after pre-processing: {features_with_nan_values}')
            
        return None
    
    def preprocess(self) -> pd.DataFrame:
        
        print(f'''\n{'-'*20} Preprocessing data  {'-'*20}''')
        # drop duplicates
        self.data.drop_duplicates(inplace=True)
        # drop useless cols
        print(f'Features to drop: {self.features_to_drop}')
        self.data.drop(columns=self.features_to_drop, axis=1)
        # fill nan values
        self._fill_nan_values()
        return self.data
        