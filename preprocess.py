import pandas as pd
import json

class PreProcessing:
    ''' Deal with nan values, drop useless columns'''
    
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.copy()
    
    def get_columns_with_nan_values(self) -> dict[str, str]:
        """
        Find columns that contain nan values.
        """
        features_with_nan_values : dict[str, float]={}
        nb_lines = self.data.shape[0]
        for c in self.data.columns :
            if self.data[c].count() < nb_lines:
                features_with_nan_values[c] = round((1 - self.data[c].count()/self.data.shape[0]) * 100, 2)
              
        return {k: str(v)+'%' for k,v in sorted(features_with_nan_values.items(), key=lambda item: item[1], reverse=True)}
                
 
    def _fill_nan_values(self, cols_median:list[str], cols_categorical:list[str]) -> None:
        """
        Fill nan values
        """
        features_with_nan_values = self.get_columns_with_nan_values()
        print(f'\nFeatures with Nan values : {json.dumps(features_with_nan_values, indent = 4)}')
        print(f'\nFeatures to fill nan values by median : {cols_median}')
        print(f'\nFeatures to fill nan values by a new category unknown : {cols_categorical}')
              
        for c in cols_median:
            self.data[c] = self.data[c].fillna((self.data[c].median()))
                        
        for c in cols_categorical:
            # fill nan values by a new category unknown(unk)
            self.data[c] = self.data[c].astype('str').replace("nan", "unk") 
        
        ## drop the value of the hand column
        self.data.p1_hand.fillna('U')
        self.data.p2_hand.fillna('U')
        self.data.reset_index(inplace=True, drop=True)
        
        features_with_nan_values = self.get_columns_with_nan_values()
        print(f'\nFeatures with Nan values after pre-processing: {features_with_nan_values}')
            
        return None
    
    def preprocess(self, cols_to_drop: list[str], cols_median:list[str], cols_categorical:list[str]) -> pd.DataFrame:
        # drop duplicates
        print(f'Features to drop: {features_to_drop}')
        self.data.drop_duplicates(inplace=True)
        # drop useless cols
        self.data.drop(columns=cols_to_drop, axis=1)
        

        # fill nan values
        self._fill_nan_values(cols_median, cols_categorical)
        return self.data
        