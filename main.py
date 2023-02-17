from ruamel.yaml import YAML
import pandas as pd
from encoder import FeaturesEncoder
from model_predict import predict
from model_train import train_models
from preprocess import PreProcessing
from sklearn.metrics import classification_report
import pickle

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


def train_test_split_per_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]:
    """split data for training and test

    Args:
        data (pd.DataFrame): preprocessed data
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]: X_train, X_test, y_train, y_test
    """

    print(f'''\n{'-'*20} Feature Engineering  {'-'*20}''')
    print(f'\nSplit Data into train and test data :')
    train_df = data[data['tourney_date'] < '2018-05-01']
    test_df = data[data['tourney_date'] >= '2018-05-01']
    print(f'\nData Train shape : {train_df.shape}, from {min(train_df["tourney_date"])} to {max(train_df["tourney_date"])}')
    print(f'\nData Test shape : {test_df.shape}, from {min(test_df["tourney_date"])} to {max(test_df["tourney_date"])}')
    
    # Target
    X_train = train_df.drop(columns=['p1_won', 'tourney_date'])
    y_train = train_df["p1_won"]
    X_test = test_df.drop(columns=['p1_won', 'tourney_date'])
    y_test = test_df["p1_won"]
    # reset_index
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    return X_train, X_test, y_train, y_test


def main():
  
    # Load config: 
    config_path = "config.yaml"

    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)
    
    data = load_data(params['data_csv_path'])
    preprocessor = PreProcessing(data, params['features_to_drop'],  \
    params['features_to_fill_by_median'], params['features_to_remove_nan_values'])
    preprocessor.preprocess()
    # split data
    X_train, X_test, y_train, y_test = train_test_split_per_time(preprocessor.data)
    # Encoding
    feature_encoder = FeaturesEncoder(params)
    # OneHotEnconder
    ohe = feature_encoder.get_onehot_encoder(X_train)
    encoded_oh_X_train = feature_encoder.transfrom_with_ohe(X_train, ohe)
    encoded_oh_X_test = feature_encoder.transfrom_with_ohe(X_test, ohe)
    # Target Encoder
    target_encoder_params = feature_encoder.get_target_encoder_params(X_train, y_train)
    encoded_X_train = feature_encoder.transform_with_target_encoder(encoded_oh_X_train, target_encoder_params)
    encoded_X_test = feature_encoder.transform_with_target_encoder(encoded_oh_X_test, target_encoder_params)
    # Train and choose the best parameters for the model
    best_model = train_models(encoded_X_train, y_train)
    # Save model
    pickle.dump(best_model, open(params['model_path']+'lightGBM_model.pkl', 'wb'))
    # Prediction
    y_pred = predict(best_model, encoded_X_train, encoded_X_test, y_train)
    print(classification_report(y_test, y_pred))

    
if __name__ == "__main__":
    main()


