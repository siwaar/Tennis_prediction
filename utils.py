import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        print('Load Data shape : ', data.shape)
        return data
    except Exception as e:
        raise OSError("No data found with the path provided in config.yaml" ) from e


def train_test_split_per_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Split data for training and test

    Args:
        data (pd.DataFrame): preprocessed data
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series]: X_train, X_test, y_train, y_test
    """

    print(f'''\n{'-'*20} Feature Engineering  {'-'*20}''')
    print(f'\nSplit Data into train and test data :')
    # sort by date before split
    data = data.sort_values('tourney_date')
    X = data.drop(columns=['p1_won'])
    y = data["p1_won"]
    # We have to not shuffle data to keep date sort
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f'\nData Train shape : {X_train.shape}, from {min(X_train["tourney_date"])} to {max(X_train["tourney_date"])}')
    print(f'\nData Test shape : {X_test.shape}, from {min(X_test["tourney_date"])} to {max(X_test["tourney_date"])}')

    X_train.drop(columns=['tourney_date'], inplace=True)
    X_test.drop(columns=['tourney_date'], inplace=True)
    # reset_index
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    
    return X_train, X_test, y_train, y_test

def display_classification_result(y_true, y_pred):
    """
    Display evaluation metrics 
    """

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)
    report = (acc, pre, rec, f1s)
    reports = [report]
    print(f'{pd.DataFrame.from_records(reports, columns=metrics)}')