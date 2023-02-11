import pandas as pd
import numpy as np


class FeatureEngineering:
    
    
    def __init__(self, data: pd.DataFrame, categorical_features: list[str], sk_kt_study_list: list[str]) -> None:
        self.data = data.copy()
        self.categorical_features = categorical_features
        self.sk_kt_study_list = sk_kt_study_list
        
    def transform(self) -> None:
        """
        Apply the feature engineering pipeline.
        """
        # Add date features : 
        self._split_date()
        # Add number of set per match:
        self._sets_per_match()
        # Add the number of game pet match:
        self._game_per_match()
        # Scale numerical data in order to deal with outliers problem:
        self._log_transform()
        # Transform seed column as categorical variable
        self._transform_seed_as_categorical()
        # create a dummies variables for categorical data
        self._process_categorical_data()
        return 
    
    def _split_date(self)-> None:
        self.data['tourney_date'] = self.data['tourney_date'].map(lambda x : str(x).split(" ")[0])
        self.data['tourney_date'] = pd.to_datetime(self.data['tourney_date'])
        self.data['day_of_week'] = self.data['tourney_date'].apply(lambda val: val.day_name())
        self.data['month'] = self.data['tourney_date'].apply(lambda val: val.month_name())
        # we need to drop tourney_date because it's not a time series problem
        self.data.drop(columns=['tourney_date'], axis=1)

        
    def _sets_per_match(self) -> None:
        """
        Get the number of set per match.
        """
        self.data["nbr_sets"] = [len(match.split()) for match in self.data["score"]]
    
    def _game_per_match(self) -> None:
        """
        Get the number of game played during a match.
        """
        games_during_all_matchs = [[g.split('-') for g in match.split()] for match in self.data["score"]]

        games_list = []
        for s in games_during_all_matchs:
            games = 0
            for g in s:
                try:
                    games += int(g[0]) + int(g[1])
                except Exception:
                    continue
            games_list.append(games)

        self.data['games_per_match'] = games_list
        self.data.drop(columns=["score"], inplace=True)
        
    def _process_categorical_data(self) -> None:
        """
        Apply get dummies for categorical data.
        """
        self.data = pd.get_dummies(self.data, columns=self.categorical_features)
        
    def _log_transform(self) -> None:
        """
        Apply log function to normalize numerical columns that need to be transformed
        """
        for col in self.sk_kt_study_list:
            try:
                self.data[f'log_{col}'] = np.log1p(self.data[col])
            except Exception:
                self.sk_kt_study_list.remove(col)
        self.data.drop(columns=self.sk_kt_study_list, inplace=True)
    
    def _transform_seed_as_categorical(self)-> None:
        """
        Consider seed as a categorical variable.
        """
        for p in [1, 2]:    
            self.data[f"p{p}_seed"] = self.data[f"p{p}_seed"].astype(str)
            self.categorical_features.append(f"p{p}_seed")
            
    
    
    