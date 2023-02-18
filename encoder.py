import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OHEncoder:
    """ Encoding with OneHot Encoding for low cardinality categorical features """
    def __init__(self, low_cardinality_categorical_features: list[str]) -> None:
        self.low_cardinality_categorical_features = low_cardinality_categorical_features
        
    def get_onehot_encoder(self, X_train: pd.DataFrame):
        """ Create OneHot encoder and fit it with data train """
        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        oh_encoder.fit(X_train[self.low_cardinality_categorical_features])
        print(f'\nNew columns after encoding low cardinality categorical features with OneHotEncoder : {oh_encoder.get_feature_names_out()}')
        return oh_encoder
    
    def transfrom_with_ohe(self, X : pd.DataFrame, oh_encoder) -> pd.DataFrame:
        """ Transform data X using the onehot encoder oh_encoder

        Args:
            X (pd.DataFrame): data to be encoded
            oh_encoder (_type_): one hot encoder

        Returns:
            pd.DataFrame: encoded data
        """
        transformed_X = pd.DataFrame(oh_encoder.transform(X[self.low_cardinality_categorical_features]))
        transformed_X.columns = oh_encoder.get_feature_names_out()
        # One-hot encoding removed index; put it back
        transformed_X.index = X.index
        # Remove categorical columns (will replace with one-hot encoding)
        other_X_cols = X.drop(self.low_cardinality_categorical_features , axis=1)
        return pd.concat([other_X_cols, transformed_X], axis=1)

class TargetEncoder:
    """ Encoding with Traget Encoding for high cardinality categorical features """
    def __init__(self) -> None:
        pass

    def get_target_encoder_params(self, X_train: pd.DataFrame, y_train : pd.Series, \
        high_cardinality_categorical_features: list[str]) -> dict[str, tuple[pd.DataFrame, float]]:
        """ Get parameters of the target encoder after fit with train set

        Args:
            X_train (pd.DataFrame): data train
            y_train (pd.Series): target

        Returns:
            _type_: parameters of target encoder
        """
        encoder_params : dict[str, tuple] = {}
        for feature in high_cardinality_categorical_features:
            averages, prior = TargetEncoder.get_target_encoder_parameters_per_feature(trn_series=X_train[feature], 
                                    target=y_train, 
                                    min_samples_leaf=100,
                                    smoothing=10)
            encoder_params[feature] = (averages, prior)
        return encoder_params

    @staticmethod
    def transform_with_target_encoder(X : pd.DataFrame, \
        encoder_params : dict[str, tuple[pd.DataFrame, float]])-> pd.DataFrame:
        """ Transform data with target encoder 

        Args:
            X (pd.DataFrame): data to be transformed
            encoder_params (_type_): parameters of encoder

        Returns:
            pd.DataFrame: transformed data
        """
        for feature, (averages, prior) in encoder_params.items():
            X[feature] = TargetEncoder.target_encode_one_feature(X[feature], averages, prior)

        return X

    @staticmethod
    def add_noise(series : pd.Series) -> pd.Series :
        noise_level = 0.01
        return series * (1 + noise_level * np.random.randn(len(series)))

    @staticmethod
    def get_target_encoder_parameters_per_feature(trn_series=None,  
                    target=None, 
                    min_samples_leaf=1, 
                    smoothing=1) -> tuple[pd.DataFrame, float]:

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
    def target_encode_one_feature(X : pd.DataFrame, averages : pd.DataFrame, prior: float) -> pd.Series:
        """ Transform one feature with Target Encoder"""
        # Apply averages 
        ft_X= pd.merge(
            X.to_frame(X.name),
            averages.reset_index().rename(columns={'index': 'p1_won', 'p1_won': 'average'}),
            on=X.name,
            how='left')['average'].rename(X.name + '_mean').fillna(prior)

        # pd.merge does not keep the index so restore it
        ft_X.index = X.index 

        return TargetEncoder.add_noise(ft_X)


   