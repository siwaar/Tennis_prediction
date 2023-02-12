import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, classification_report
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
    print(f'Best model based on f1 score :{best_m} with score : {round(best_sc*100,2)} %')
    return get_best_model(X_train, y_train, models[best_m])


def score_classifier(dataset, classifier, labels):
    
    """
    Performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    Inputs :
        - dataset : the dataset to work on
        - classifier : the classifier to use
        - labels : the labels used for training and validation
    :return:
    """
    n_splits = 3
    kf = KFold(n_splits=n_splits, random_state=50, shuffle=True)
    confusion_matrices = []
    recalls = []
    precisions = []
    accuracies = []
    f1_scores = []
    
    for training_ids, test_ids in kf.split(dataset):
        
        if type(dataset) == pd.DataFrame:
            
            training_set = dataset.loc[training_ids]
            training_labels = labels.loc[training_ids]

            test_set = dataset.loc[test_ids]
            test_labels = labels.loc[test_ids]
            
        elif type(dataset) == np.ndarray: 
            
            training_set = dataset[training_ids]
            training_labels = labels[training_ids]

            test_set = dataset[test_ids]
            test_labels = labels[test_ids]
            
        
        classifier.fit(training_set, training_labels)
        
        predicted_labels = classifier.predict(test_set)
        
        confusion_matrices.append(confusion_matrix(test_labels, predicted_labels))
        recalls.append(recall_score(test_labels, predicted_labels))
        precisions.append(precision_score(test_labels, predicted_labels))
        f1_scores.append(f1_score(test_labels, predicted_labels))
        accuracies.append(accuracy_score(test_labels, predicted_labels))
    
    
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
    print("\n############################\n")
    return f1_sc

def get_best_model(X_train , y_train, classifier):
    clf = GridSearchCV(
            estimator=classifier(),
            param_grid={'num_leaves': (15, 30, 45),
                        'max_depth': (-1, 5, 10, 20),
                        'learning_rate': (0.05, 0.1, 0.2, 0.4),
                        'n_estimators': (25, 50, 100, 200)
                        },
            scoring='f1',
            cv=3,
            n_jobs=3,
            verbose=1,
            refit=True)

    clf.fit(X_train,y_train)
    return classifier(**clf.best_params_)