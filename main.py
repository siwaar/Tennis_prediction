from ruamel.yaml import YAML
import pandas as pd
from feature_engineering import FeatureEngineering
from model_predict import predict
from model_train import train_models
from preprocess import PreProcessing
from sklearn.model_selection import train_test_split

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


def preprocess_data(params: dict[str, str]) -> pd.DataFrame:
    """
        Load and process data

    Args:
        params (dict[str, str]): a dictionary containing
        data path and selected columns to deal with the task.

    Returns:
        pd.DataFrame: preprocessed data
    """
    # Load data
    data = load_data(params['data_csv_path'])
    # Preprocess data
    preprocessing = PreProcessing(data, params['features_to_drop'],  \
        params['features_to_fill_by_median'], params['features_to_fill_by_new_category'])
    preprocessing.preprocess()
    print(f'\nData shape after preprocessing : {preprocessing.data.shape}')

    return preprocessing.data

def apply_feature_engineering(data: pd.DataFrame, params: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]:
    """Apply feature engineering and split data for training and test

    Args:
        data (pd.DataFrame): preprocessed data
        params (dict[str, str]): a dictionary containing selected columns for a spacial feature engineering.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]: X_train, X_test, y_train, y_test
    """

    print(f'''\n{'-'*20} Feature Engineering  {'-'*20}''')
    y = data["p1_won"]
    X = data.drop(columns=['p1_won'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)

    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    # Feature engineering
    feature_engineering_train = FeatureEngineering(X_train, params, y_train)
    X_train = feature_engineering_train.transform()
    print(f'\nX_train shape after feature engineering : {X_train.shape}')
    feature_engineering_test = FeatureEngineering(X_test, params, y_test)
    X_test = feature_engineering_test.transform()
    print(f'\nX_test shape after feature engineering : {X_test.shape}')
    return X_train, X_test, y_train, y_test


def main():
    # Load config: 
    config_path = "config.yaml"

    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)
    
    # Load and preprocess data 
    data = preprocess_data(params)
    # Feature Engineering
    X_train, X_test, y_train, y_test = apply_feature_engineering(data, params)
    # Train and choose the best model
    best_model = train_models(X_train, y_train)
    # Save model
    pickle.dump(best_model, open(params['model_path']+'lightGBM_model.pkl', 'wb'))
    # Prediction
    y_pred = predict(best_model, X_train, X_test, y_train)
    print(classification_report(y_test, y_pred))

    
if __name__ == "__main__":
    main()


