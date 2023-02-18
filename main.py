from ruamel.yaml import YAML
import pandas as pd
from encoder import FeaturesEncoder
from preprocess import PreProcessing, load_data
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMClassifier

def train_models(X_train , y_train) :
    """ Train and finetune model  """

    print(f'''\n{'-'*20} Train LightGBM model  {'-'*20}''')
    model = LGBMClassifier
    model().fit(X_train, y_train)
    
    print(f'''\n{'-'*20} Finetune LightGBM model  {'-'*20}''')
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits)
    model_tuned = GridSearchCV(
            estimator=model(),
            param_grid={'num_leaves': (15, 31, 45),
                        'max_depth': (-1, 5, 10),
                        'learning_rate': (0.05, 0.1, 0.2),
                        'n_estimators': (25, 50, 100)
                        },
            scoring='f1',
            cv=tscv,
            n_jobs=3,
            verbose=1,
            refit=True
            )

    model_tuned.fit(X_train,y_train)
    # printing the best parameters
    print(f"Best parameters {model_tuned.best_params_}")
    best_model = model(**model_tuned.best_params_)
    best_model.fit(X_train, y_train)
    return best_model


def train_test_split_per_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]:
    """split data for training and test

    Args:
        data (pd.DataFrame): preprocessed data
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]: X_train, X_test, y_train, y_test
    """

    print(f'''\n{'-'*20} Feature Engineering  {'-'*20}''')
    print(f'\nSplit Data into train and test data :')
    data = data.sort_values('tourney_date')
    X = data.drop(columns=['p1_won'])
    y = data["p1_won"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f'\nData Train shape : {X_train.shape}, from {min(X_train["tourney_date"])} to {max(X_train["tourney_date"])}')
    print(f'\nData Test shape : {X_test.shape}, from {min(X_test["tourney_date"])} to {max(X_test["tourney_date"])}')
    
    X_train = X_train.drop(columns=['tourney_date'])
    X_test = X_test.drop(columns=['tourney_date'])
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
    # save onehot encoder
    pickle.dump(ohe, open(params['onehot_encoder_path'], 'wb'))
    encoded_oh_X_train = feature_encoder.transfrom_with_ohe(X_train, ohe)
    encoded_oh_X_test = feature_encoder.transfrom_with_ohe(X_test, ohe)
    # Target Encoder
    target_encoder_params = feature_encoder.get_target_encoder_params(X_train, y_train)
    # save target encoder
    pickle.dump(target_encoder_params, open(params['target_encoder_path'], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # The advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster
    encoded_X_train = feature_encoder.transform_with_target_encoder(encoded_oh_X_train, target_encoder_params)
    encoded_X_test = feature_encoder.transform_with_target_encoder(encoded_oh_X_test, target_encoder_params)
    # Train and choose the best parameters for the model
    best_model = train_models(encoded_X_train, y_train)
    # Save model
    pickle.dump(best_model, open(params['model_path'], 'wb'))
    # Prediction
    y_pred = best_model.predict(encoded_X_test)
    print(classification_report(y_test, y_pred))

    
if __name__ == "__main__":
    main()


