import pandas as pd 
from sklearn.model_selection import train_test_split

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