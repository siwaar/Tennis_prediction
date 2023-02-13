import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, categorical_features: list[str], sk_kt_cols: list[str],\
        categorical_features_target_encoding:list[str], target: pd.DataFrame) -> None:
        self.data = data.copy()
        self.categorical_cols = categorical_features
        self.sk_kt_cols = sk_kt_cols
        self.categorical_features_target_encoding = categorical_features_target_encoding
        self.target = target
        
    def _split_date(self) -> None:
        """
        Extract the year, the month and the day from the column tourney date.
        """
        
        self.data["day"] = self.data["tourney_date"]%100
        self.data["month"] = (self.data["tourney_date"]//100)%100
        self.data["year"] = self.data["tourney_date"]//10000
        #self.data['day_of_week'] = pd.to_datetime(self.data['tourney_date'].map(lambda x : str(x))).apply(lambda val: val.day_name())
        #self.categorical_cols.append('day_of_week')
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


    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    @staticmethod
    def target_encode(trn_series=None,  
                    target=None, 
                    min_samples_leaf=1, 
                    smoothing=1,
                    noise_level=0):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior  
        """ 
        assert len(trn_series) == len(target)
        temp = pd.concat([trn_series, target], axis=1)
        
        # Compute target mean 
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        
        # Apply average function to all target data
        prior = target.mean()
        
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index 

        return FeatureEngineering.add_noise(ft_trn_series, noise_level)

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
        # Apply target encoding
        for feature in self.categorical_features_target_encoding:
            feature_encoded = FeatureEngineering.target_encode(self.data[feature],
                                target=self.target, 
                                min_samples_leaf=100,
                                smoothing=10,
                                noise_level=0.01)
            self.data[feature] = feature_encoded
        # Transform seed column as categorical variable
        self._transform_seed_as_categorical()
        # create a dummies variables for categorical data
        self._process_categorical_data()
        return self.data