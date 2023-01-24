from time import time
from print_time import print_time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


# Quick tryouts for several models
def try_models(models, X, y, plotting=True):
    scores_df = []
    n_models = len(models)
    for i, model in enumerate(models):
        print(f'({i+1}/{n_models}) {model.__class__.__name__}')

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

    scores_df = pd.DataFrame(scores_df, columns=['Model', 'Time', 'Score']).set_index('Model')
    scores_df = scores_df.sort_values(by='Score')

    if plotting:
        ax = scores_df.Score.plot(kind='barh')
        for bar, score, time_taken in zip(ax.patches, scores_df.Score.to_list(), scores_df.Time.to_list()):
            ax.text(0.25, bar.get_y() + bar.get_height() / 2 - 0.1, f'%{score * 100:.4f} (Time: {time_taken})',
                    color='white', size=20)


# Geting cross validation score and predictions
def cross_validation(model_to_call, pars, X, y, skf, scoring,
                     verbose=0, submit_pred=False, X_submit=None, method='predict_proba', submit_file=False,
                     sample_submission=None, target='target'):
    models = []
    scores = []
    y_submit_list = []
    submit_pred = True if submit_file else submit_pred
    for i, (train_ind, val_ind) in enumerate(skf.split(X, y)):
        X_train, X_val = X.loc[train_ind], X.loc[val_ind]
        y_train, y_val = y.loc[train_ind], y.loc[val_ind]

        model = model_to_call(**pars)

        if model.__class__.__name__ == 'CatBoostClassifier':
            model.fit(X_train.values, y_train, early_stopping_rounds=100,
                      eval_set=[(X_val.values, y_val)], verbose=0)

        else:
            model.fit(X_train, y_train)

        if method == 'predict_proba':
            y_pred = model.predict_proba(X_val)[:, 1]
        else:
            y_pred = model.decision_function(X_val)
        score = scoring(y_val, y_pred)

        if verbose > 1:
            print(f'Fold {i + 1:02}: Score={score}')
        scores.append(score)
        models.append(model)

        if submit_pred:
            if method == 'predict_proba':
                y_submit_list.append(model.predict_proba(X_submit)[:, 1])
            else:
                y_submit_list.append(model.decision_function(X_submit))

    if verbose > 0:
        print('_' * 25, '\n', f'Mean: {np.mean(scores)}')

    if submit_pred:
        y_submit = np.stack(y_submit_list).mean(0)
    else:
        y_submit = None

    if submit_file:
        sample_submission[target] = y_submit
        sample_submission.to_csv('submission.csv', index=False)

    return models, scores, np.mean(scores), y_submit
