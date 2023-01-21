from time import time
from print_time import print_time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


# Quick tryouts for several models
def try_models(models, X, y):
    scores_df = []
    n_models = len(models)
    for i, model in enumerate(models):
        print(f'({i + 1}/{n_models}) {model.__class__.__name__}')

        before = time()
        try:
            y_pred = cross_val_predict(model, X, y, method='predict_proba', cv=5, n_jobs=-1)[:, 1]
        except AttributeError:
            y_pred = cross_val_predict(model, X, y, method='decision_function', cv=5, n_jobs=-1)
        time_taken = print_time(before, returning=True)

        score = roc_auc_score(y, y_pred)
        print(f'Score={score}')

        scores_df.append([model.__class__.__name__, time_taken, score])
        print()

    return pd.DataFrame(scores_df, columns=['Model', 'Time', 'Score'])


# Geting cross validation score and predictions
def cross_validation(model_to_call, pars, X, y, skf, scoring,
                     verbose=0, submit_file=False, X_submit=None, sample_submission=None,
                     target='target'):
    models = []
    scores = []
    y_submit_list = []

    for i, (train_ind, val_ind) in enumerate(skf.split(X, y)):
        X_train, X_val = X.loc[train_ind], X.loc[val_ind]
        y_train, y_val = y.loc[train_ind], y.loc[val_ind]

        model = model_to_call(**pars)

        if model.__class__.__name__ == 'CatBoostClassifier':
            model.fit(X_train.values, y_train, early_stopping_rounds=100,
                      eval_set=[(X_val.values, y_val)], verbose=0)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:, 1]
        score = scoring(y_val, y_pred)

        if verbose > 1:
            print(f'Fold {i + 1:02}: Score={score}')
        scores.append(score)
        models.append(model)

        if submit_file:
            y_submit_list.append(model.predict_proba(X_submit)[:, 1])

    if verbose > 0:
        print('_' * 25, '\n', f'Mean: {np.mean(scores)}')

    if submit_file:
        y_submit = np.stack(y_submit_list).mean(0)
        sample_submission[target] = y_submit
        sample_submission.to_csv('submission.csv', index=False)

    return models, scores
