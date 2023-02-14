import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, params: dict[str, str], target: pd.DataFrame) -> None:
        """ Apply feature engineering 

        Args:
            data (pd.DataFrame): preprocessed data
            params (dict[str, str]): a dictionary containing selected columns for a 
            spacial feature engineering.
            target (pd.DataFrame): ground truth of the training set
        """
        self.data = data.copy()
        self.categorical_cols = params['categorical_features_dummies']
        self.categorical_features_target_encoding = params['categorical_features_target_encoding']
        self.redundant_features = params["redundant_features"]
        self.cols_to_scale = params["features_to_scale"]
        self.target = target
        
    def _split_date(self) -> None:
        """  Extract the year, the month and the day from the column tourney date.
        """
        
        self.data["day"] = self.data["tourney_date"]%100
        self.data["month"] = (self.data["tourney_date"]//100)%100
        self.data["year"] = self.data["tourney_date"]//10000
        datetime_feature = pd.to_datetime(self.data['tourney_date'].map(lambda x : str(x)))
        self.data['day_of_week'] = datetime_feature.apply(lambda val: val.day_name())
        self.categorical_cols.append('day_of_week')
        self.data['is_weekend'] = datetime_feature.map(lambda x : 1 if x.weekday() >= 5 else 0 )
        self.data.drop(columns=["tourney_date"], inplace=True)
        
    def _sets_per_match(self) -> None:
        """
        Get the number of sets per match.
        """
        self.data["nbr_sets"] = [len(match.split()) for match in self.data["score"]]
    
    def _game_per_match(self) -> None:
        """
        Get the number of games played during a match.
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
        
    def _scale_numerical_features(self) -> None:
        """ Scale numerical features """
        features = self.data[self.cols_to_scale]
        scaler = RobustScaler().fit(features.values)
        self.data[self.cols_to_scale] = scaler.transform(features.values)
         
    
    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def _target_encode(self, min_samples_leaf, smoothing, noise_level) -> None:
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior  
        """ 
        for feature in self.categorical_features_target_encoding:
            temp = pd.concat([self.data[feature], self.target], axis=1)
            
            # Compute target mean 
            averages = temp.groupby(by=self.data[feature].name)[self.target.name].agg(["mean", "count"])
            
            # Compute smoothing
            smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            
            # Apply average function to all target data
            prior = self.target.mean()
            
            # The bigger the count the less full_avg is taken into account
            averages[self.target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
            averages.drop(["mean", "count"], axis=1, inplace=True)
            
            # Apply averages to trn and tst series
            ft_trn_series = pd.merge(
                self.data[feature].to_frame(self.data[feature].name),
                averages.reset_index().rename(columns={'index': self.target.name, self.target.name: 'average'}),
                on=self.data[feature].name,
                how='left')['average'].rename(self.data[feature].name + '_mean').fillna(prior)
            
            # pd.merge does not keep the index so restore it
            ft_trn_series.index = self.data[feature].index 
            self.data[feature] = FeatureEngineering.add_noise(ft_trn_series, noise_level)


    def transform(self) -> pd.DataFrame:
        """ Apply the feature engineering pipeline.

        Returns:
            pd.DataFrame: processed data
        """
        # Add date features : 
        self._split_date()
        # Add number of sets per match:
        self._sets_per_match()
        # Add the number of games per match:
        self._game_per_match()
        # Apply target encoding
        self._target_encode(min_samples_leaf=100, smoothing=10, noise_level=0.01)
        # create a dummies variables for categorical data
        self._process_categorical_data()
        # scale numerical features
        self._scale_numerical_features()
        # remove redundant features
        self.data.drop(columns=self.redundant_features, inplace=True)

        return self.data