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