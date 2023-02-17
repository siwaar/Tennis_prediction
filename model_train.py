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
    X_train= X_train.copy()
    X_train['date'] = pd.to_datetime(X_train[['year', 'month', 'day']])
    X_train = X_train.sort_values('date')
    X_train.drop(columns=['date'], inplace=True)

    model_tuned = GridSearchCV(
            estimator=model(),
            param_grid={'num_leaves': (15, 30, 45),
                        'max_depth': (-1, 5, 10, 20),
                        'learning_rate': (0.05, 0.1, 0.2, 0.4),
                        'n_estimators': (25, 50, 100, 200)
                        },
            scoring='f1',
            cv=tscv,
            n_jobs=3
            )

    model_tuned.fit(X_train,y_train)
    # printing the best parameters
    print(f"Best parameters {model_tuned.best_params_}")
    return model(**model_tuned.best_params_)