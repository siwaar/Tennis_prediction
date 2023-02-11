import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, categorical_features: list[str], sk_kt_cols: list[str]) -> None:
        self.data = data.copy()
        self.categorical_cols = categorical_features
        self.sk_kt_cols = sk_kt_cols
        
    def _split_date(self) -> None:
        """
        Extract the year, the month and the day from the column tourney date.
        """
        self.data["day"] = self.data["tourney_date"]%100
        self.data["month"] = (self.data["tourney_date"]//100)%100
        self.data["year"] = self.data["tourney_date"]//10000
        self.data.drop(columns=["tourney_date"], inplace=True)
    
    def _importance_tourney(self) -> None:
        """
        Set A new column which represent the importance of the tourney.
        """
        importance_dict = {"G":6, "M": 5, "A": 4, "C": 4, "S": 3, "F" : 2, "D": 1}
        importance = [importance_dict[k] for k in self.data["tourney_level"].tolist()]
        self.data["tourney_importance"] = importance
        
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
        self.data = pd.get_dummies(self.data, columns=self.categorical_cols)
        
    def _log_transform(self) -> None:
        """
        Apply log function to normalize numerical columns that need to be transformed
        """
        for col in self.sk_kt_cols:
            try:
                self.data[f'log_{col}'] = np.log1p(self.data[col])
            except Exception:
                self.sk_kt_cols.remove(col)
        self.data.drop(columns=self.sk_kt_cols, inplace=True)
    
    def _transform_seed_as_categorical(self):
        """
        Consider seed as a categorical variable.
        """
        for p in [1, 2]:    
            self.data[f"p{p}_seed"] = self.data[f"p{p}_seed"].astype(str)

    def transform(self) -> pd.DataFrame:
        """
        Apply the feature engineering pipeline.
        """
        # Add date features : 
        self._split_date()
        # Add importance tourney feature:
        self._importance_tourney()
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
        return self.data