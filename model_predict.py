import pandas as pd


def predict(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
    """
    input:
        - model : model after tuning hyperparameters
        - X_train : training set dataframe after feature engineering
        - y_train : ground truth of the trianing set
        - X_test : test set dataframe after feature engineering
    output : 
        
    """
    print(f'''\n{'-'*20} Model prediction  {'-'*20}''')
    def intersection(lst1: list[str], lst2: list[str]) -> list[str]:
        return list(set(lst1) & set(lst2))

    X_tr = X_train.copy()
    X_ts = X_test.copy()

    # Remove duplicate columns :
    X_tr = X_tr.loc[:,~X_tr.columns.duplicated()]
    X_ts = X_ts.loc[:,~X_ts.columns.duplicated()]
    
    # Get columns intersection : 
    cols = intersection(list(X_tr.columns), list(X_ts.columns))
    
    X_tr = X_tr[cols]
    X_ts = X_ts[cols]
    
    assert list(X_tr.columns) == list(X_ts.columns)
    
    model.fit(X_tr, y_train)
    return model.predict(X_ts)