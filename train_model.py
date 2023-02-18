from time import time
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
np.random.seed(42)


def train_models(X_train : pd.DataFrame , y_train: pd.DataFrame) -> LGBMClassifier:
    """ Train and finetune model

        Args:
            X_train (pd.DataFrame): Data Train
            y_train (pd.DataFrame): Data target

        Returns:
            LGBMClassifier: trained and fine tuned model
    """  

    print(f'''\n{'-'*20} Train LightGBM model with default hyperparameters {'-'*20}''')
    model = LGBMClassifier
    start_train = time()
    model().fit(X_train, y_train)
    end_train = time()
    print(f'''\nLightGBM was trained during : {end_train - start_train} sec''')

    # Fine Tunning 
    print(f'''\n{'-'*20} Finetune model with Time Series Cross Validation {'-'*20}''')
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits)
    cv = GridSearchCV(
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

    cv.fit(X_train,y_train)
    print(f"\nBest parameters : {cv.best_params_}")
    best_model = model(**cv.best_params_)
    print(f"\nTrain LightGBM model with best parameters : {cv.best_params_}")
    best_model.fit(X_train, y_train)
    return best_model   