import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class FeaturesEncoder:
    def __init__(self, params: dict[str, str]) -> None:
        """ Apply feature engineering 
        Args:
            data (pd.DataFrame): preprocessed data
            params (dict[str, str]): a dictionary containing selected columns for a 
            spacial feature engineering.
            target (pd.DataFrame): ground truth of the training set
        """
        self.low_cardinality_categorical_features = params['low_cardinality_categorical_features']
        self.high_cardinality_categorical_features = params['high_cardinality_categorical_features']
        
    def get_onehot_encoder(self, X_train: pd.DataFrame):
        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        oh_encoder.fit(X_train[self.low_cardinality_categorical_features])
        print(f'\nNew columns after encoding low cardinality categorical features with OneHotEncoder : {oh_encoder.get_feature_names_out()}')
        return oh_encoder
    
    def transfrom_with_ohe(self, X:pd.DataFrame, oh_encoder) -> pd.DataFrame:
        transformed_X = pd.DataFrame(oh_encoder.transform(X[self.low_cardinality_categorical_features]))
        transformed_X.columns = oh_encoder.get_feature_names_out()
        # One-hot encoding removed index; put it back
        transformed_X.index = X.index
        # Remove categorical columns (will replace with one-hot encoding)
        other_X_cols = X.drop(self.low_cardinality_categorical_features , axis=1)
        return pd.concat([other_X_cols, transformed_X], axis=1)

    def get_target_encoder_params(self, X_train: pd.DataFrame, y_train):
        encoder_params : dict[str, tuple] = {}
        for feature in self.high_cardinality_categorical_features:
            averages, prior = FeaturesEncoder.get_target_encoder_parameters_per_feature(trn_series=X_train[feature], 
                                    target=y_train, 
                                    min_samples_leaf=100,
                                    smoothing=10)
            encoder_params[feature] = (averages, prior)
        return encoder_params

    def transform_with_target_encoder(self, X:pd.DataFrame, encoder_params)-> pd.DataFrame:
        for feature, (averages, prior) in encoder_params.items():
            X[feature] = FeaturesEncoder.target_encode_one_feature(X[feature], averages, prior)

        return X


    @staticmethod
    def add_noise(series):
        noise_level = 0.01
        return series * (1 + noise_level * np.random.randn(len(series)))
    
    @staticmethod
    def get_target_encoder_parameters_per_feature(trn_series=None,  
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1):

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
        
        return averages, prior
    
    @staticmethod
    def target_encode_one_feature(X, averages, prior):
        # Apply averages 
        ft_X= pd.merge(
            X.to_frame(X.name),
            averages.reset_index().rename(columns={'index': 'p1_won', 'p1_won': 'average'}),
            on=X.name,
            how='left')['average'].rename(X.name + '_mean').fillna(prior)

        # pd.merge does not keep the index so restore it
        ft_X.index = X.index 

        return  FeaturesEncoder.add_noise(ft_X)


   