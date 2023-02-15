import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train) -> None:
    """ Train models """

    print(f'''\n{'-'*20} Train models and choose the best one  {'-'*20}''')
    models = {'LGBMClassifier': LGBMClassifier,
       'XGBClassifier':XGBClassifier,
    }

    best_m =''
    best_sc = 0
    for model in models:
        print(f'\nModel : {model}')
        score = score_classifier(X_train, models[model](), y_train)
        if score > best_sc:
            best_sc = score
            best_m = model
    print(f'Best model based on f1 score :{best_m}')
    return get_best_model(X_train, y_train, models[best_m])


def score_classifier(features_set: pd.DataFrame, classifier, labels: pd.Series) -> float:
    """
    Performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    Inputs :
        - dataset : the dataset to work on
        - classifier : the classifier to use
        - labels : the labels used for training and validation
    :return:
    """
    
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits)
    confusion_matrices = []
    recalls = []
    precisions = []
    accuracies = []
    f1_scores = []
    fold = 0
    # Sort train data by date
    dataset= features_set.copy()
    dataset['date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
    dataset = dataset.sort_values('date')

    # Iterate through each split
    for train_index, test_index in tscv.split(dataset):
        
        cv_train_set, cv_test_set = dataset.iloc[train_index], dataset.iloc[test_index]
        cv_train_label, cv_test_label = labels.iloc[train_index], labels.iloc[test_index]

    
        print('Fold :', fold)
        print('Train date range: from {} to {}'.format(cv_train_set.date.min(), cv_train_set.date.max()))
        print('Test date range: from {} to {}\n'.format(cv_test_set.date.min(), cv_test_set.date.max()))
        fold += 1
        cv_train_set.drop(columns=['date'], inplace=True)
        cv_test_set.drop(columns=['date'], inplace=True)
        classifier.fit(cv_train_set, cv_train_label)
        
        predicted_labels = classifier.predict(cv_test_set)
        
        confusion_matrices.append(confusion_matrix(cv_test_label, predicted_labels))
        recalls.append(recall_score(cv_test_label, predicted_labels))
        precisions.append(precision_score(cv_test_label, predicted_labels))
        f1_scores.append(f1_score(cv_test_label, predicted_labels))
        accuracies.append(accuracy_score(cv_test_label, predicted_labels))
    
    recall = np.mean(recalls) 
    precision = np.mean(precisions)
    accuracy = np.mean(accuracies)
    f1_sc = np.mean(f1_scores)
    confusion_mat = np.mean(confusion_matrices, axis=0)
    
    print(f"Recall = {recall}")
    print(f"Precision = {precision}")
    print(f"Accuracy = {accuracy}")
    print(f"f1 score = {f1_sc}")
    print(f"\nConfusion matrix : \n\n {confusion_mat}")
    print(f"\n{'-'*20}\n")
    return f1_sc

def get_best_model(X_train , y_train, classifier):
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits)
    X_train= X_train.copy()
    X_train['date'] = pd.to_datetime(X_train[['year', 'month', 'day']])
    X_train = X_train.sort_values('date')
    X_train.drop(columns=['date'], inplace=True)

    clf = GridSearchCV(
            estimator=classifier(),
            param_grid={'num_leaves': (15, 30, 45),
                        'max_depth': (-1, 5, 10, 20),
                        'learning_rate': (0.05, 0.1, 0.2, 0.4),
                        'n_estimators': (25, 50, 100, 200)
                        },
            scoring='f1',
            cv=tscv,
            n_jobs=3,
            verbose=1,
            refit=True)

    clf.fit(X_train,y_train)
    return classifier(**clf.best_params_)