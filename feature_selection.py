from time import time
from random import shuffle
from print_time import print_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cross_val_predict, roc_auc_score


# Converting an integer to a binary representation
def binary(x, n_bits, ret_type='bool_list'):
    x = format(x, '#0' + str(n_bits + 2) + 'b')[2:]

    if ret_type == 'str':
        return x

    if ret_type == 'str_list':
        return list(x)

    if ret_type == 'int_list':
        return [int(i) for i in list(x)]

    return [bool(int(i)) for i in list(x)]


# Sequential feature selection
def feature_select(model_to_call, pars, X, y, dont_touch=[], scoring=roc_auc_score):
    cols = list(X.columns)
    cols = list(set(cols) - set(dont_touch))
    dropped = []
    model = model_to_call(**pars)

    try:
        y_pred = cross_val_predict(model, X, y, method='predict_proba', cv=5)[:, 1]
    except AttributeError:
        y_pred = cross_val_predict(model, X, y, method='decision_function', cv=5, random_state=69)

    global_best_score = scoring(y, y_pred)

    while True:
        n_dropped = len(dropped)
        print(f'We now have score of {global_best_score:.5f} with {n_dropped} columns dropped.')
        print(f'Searching for a column to drop...')
        print('_' * 25)
        model = model_to_call(**pars)
        local_best_score = -1000
        choice = None
        delta_score = -1000
        n_remaining = len(cols)

        for i, col in enumerate(cols):
            temp_dropped = dropped + [col]

            before = time()
            try:
                y_pred = cross_val_predict(model, X.drop(temp_dropped, axis=1), y, method='predict_proba', cv=5)[:, 1]

            except AttributeError:
                y_pred = cross_val_predict(model, X.drop(temp_dropped, axis=1), y, method='decision_function', cv=5)
            time_taken = print_time(before, printing=False, returning=True)

            score = scoring(y, y_pred)
            temp_delta_score = score - global_best_score
            print(f'[{i+1:2}/{n_remaining:2}] Dropping {col:30}, time-taken:{time_taken:5}, score={score:.5f}, delta-score={temp_delta_score:.5f}')

            if score > local_best_score:
                local_best_score = score
                choice = col
                delta_score = temp_delta_score

        if delta_score > 0:
            dropped.append(choice)
            cols = list(set(cols) - set(dropped))
            global_best_score = local_best_score
            print('_' * 25)
            print(f'column {choice} has been dropped.')
            continue

        print('No high score observed. Procedure has ended.')
        return dropped, global_best_score


# Grid feature selection
def grid_feature_selection(model_to_call, params, X, y, possible_bad_cols, scoring,
                           method='predict_proba', next_query=0.1, cv=5, dont_ask=False, plotting=True):
    n_cols = len(possible_bad_cols)
    n_possibilities = 2 ** n_cols
    next_query = next_query if next_query > 1 else int(n_possibilities * next_query)
    cases_list = list(range(n_possibilities))
    shuffle(cases_list)
    possible_bad_cols = np.array(possible_bad_cols)

    scores_dict = {col: {'With': [], 'Without': []} for col in possible_bad_cols}
    cases_dict = {'Case': [], 'Score': []}
    best_score = -1000
    best_case = None
    iterations_since_last_high = 0

    print(f'There are {n_possibilities} possibilities to check.')
    if next_query >= n_possibilities:
        print('Procedure will be continuing till the end.')
    elif dont_ask:
        print(f'Procedure will be continuing till iteration {next_query}.')
    else:
        print(f'You will be asked if you want to continue after {next_query} iterations.')
    print('_' * 25)

    for i, case in enumerate(cases_list):
        model = model_to_call(**params)
        binary_case = binary(case, n_cols)
        anti_case = [not i for i in binary_case]

        with_us = possible_bad_cols[binary_case]
        not_with_us = possible_bad_cols[anti_case]

        before = time()
        y_pred = cross_val_predict(model, X.drop(not_with_us, axis=1), y, method=method, cv=cv, n_jobs=8)
        time_taken = print_time(before, returning=True, printing=False)

        y_pred = y_pred[:, 1] if method == 'predict_proba' else y_pred
        score = scoring(y, y_pred)

        for col in with_us:
            scores_dict[col]['With'].append(score)
        for col in not_with_us:
            scores_dict[col]['Without'].append(score)

        cases_dict['Case'].append(case)
        cases_dict['Score'].append(score)

        high_score = 'HS' if score > best_score else '__'
        if score > best_score:
            best_score = score
            best_case = case
            iterations_since_last_high = 0
        else:
            iterations_since_last_high += 1

        print(f'({i + 1:03}/{n_possibilities}) Case:{case:03} Time-Taken:{time_taken} Score:{score:.5f} {high_score}')

        if i + 1 == next_query:
            print('_' * 25)
            print(f'\nModel has achived score of {best_score} until now.')
            if dont_ask or i + 1 == n_possibilities:
                break

            print(f'It has been {iterations_since_last_high} iterations since the last high score.')
            print('Do you want to continue?')
            s = 'Enter 0 if you want to exit, or enter a number of iterations you want to add: '

            try:
                user_input = int(input(s))
            except ValueError:
                user_input = 0

            new_query = user_input + i + 1
            if new_query >= n_possibilities:
                print('Procedure is being continued till the end.')

            elif new_query > i + 1:
                next_query = new_query
                print(f'Procedure is being continued, You will be asked again in iteration {next_query}.')
                print('_' * 25)

            else:
                print('Procedure is ended.')
                break

    cases_df = pd.DataFrame(cases_dict).sort_values(by='Score', ascending=False).reset_index().drop('index', axis=1)
    anti_best_case = [not i for i in binary(best_case, n_cols)]
    best_case_dropped_cols = possible_bad_cols[anti_best_case]

    if plotting:
        plot_take_or_drop(scores_dict)

    return best_case, best_score, best_case_dropped_cols, cases_df, scores_dict


# Plotting the data returned from grid_feature_selection
def plot_take_or_drop(scores_dict):
    n_cols = len(scores_dict)
    cols = list(scores_dict)
    plt.rcParams['figure.figsize'] = (np.clip(n_cols, a_min=7, a_max=20), 18)
    with_us = [np.mean(scores_dict[col]['With']) for col in cols]
    not_with_us = [np.mean(scores_dict[col]['Without']) for col in cols]
    minimum_value = min(min(with_us), min(not_with_us)) * 0.999
    maximum_value = max(max(with_us), max(not_with_us)) * 1.001
    diffs = list(np.array(with_us) - np.array(not_with_us))
    plt.subplots(2, 1)

    plt.subplot(2, 1, 1)
    x_axis = np.arange(n_cols)
    plt.bar(x_axis - 0.2, with_us, 0.4, color='g', label='Average score when keeping feature')
    plt.bar(x_axis + 0.2, not_with_us, 0.4, color='r', label='Average score when dropping feature')
    plt.xticks(x_axis, cols, rotation=90)
    plt.ylim(minimum_value, maximum_value)
    plt.title('Average score with and without features')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(cols, diffs, marker='.', markersize=20)
    plt.axhline(y=0, linewidth=2, color='k')
    plt.xticks(rotation=90)
    plt.title('Difference between average scores (Keeping score avg - Dropping score avg)')

    plt.show()
